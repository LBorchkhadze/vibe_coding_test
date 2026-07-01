import asyncio
import io
import json
import os

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from starlette.datastructures import UploadFile

app = FastAPI(title="Reeds Jobs API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GREENHOUSE_BOARDS = [
    "riskified",
    "fireblocks",
    "pagayais",
    "gongio",
    "lightricks",
    "similarweb",
    "melio",
    "wizinc",
    "yotpo",
    "catonetworks",
]
GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"

# --- Gemini ranking configuration ---
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)
# How many jobs to score per Gemini request, and how many requests to run at once.
RANK_BATCH_SIZE = int(os.environ.get("RANK_BATCH_SIZE", "25"))
RANK_CONCURRENCY = int(os.environ.get("RANK_CONCURRENCY", "5"))


async def fetch_board(client: httpx.AsyncClient, token: str) -> list[dict]:
    """Fetch all jobs for a single Greenhouse board and tag them with the company."""
    response = await client.get(GREENHOUSE_URL.format(token=token))
    response.raise_for_status()
    data = response.json()
    jobs = []
    for job in data.get("jobs", []):
        location = job.get("location") or {}
        jobs.append(
            {
                "title": job.get("title"),
                "location": location.get("name"),
                "apply_url": job.get("absolute_url"),
                "company": token,
            }
        )
    return jobs


async def fetch_all_jobs(client: httpx.AsyncClient) -> list[dict]:
    """Fetch and combine jobs from every configured Greenhouse board concurrently."""
    results = await asyncio.gather(
        *(fetch_board(client, token) for token in GREENHOUSE_BOARDS),
        return_exceptions=True,
    )

    jobs: list[dict] = []
    for token, result in zip(GREENHOUSE_BOARDS, results):
        if isinstance(result, Exception):
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch jobs for board '{token}': {result}",
            )
        jobs.extend(result)
    return jobs


@app.get("/jobs")
async def get_jobs() -> dict:
    """Fetch jobs from all configured Greenhouse boards concurrently and combine them."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        jobs = await fetch_all_jobs(client)
    return {"count": len(jobs), "jobs": jobs}


# Field names the frontend might use for the role and the CV (case-insensitive).
_ROLE_FIELDS = {"role", "target_role", "targetrole", "job_role", "jobrole", "position"}
_CV_TEXT_FIELDS = {"cv", "resume", "cv_text", "resume_text", "text"}


def _pdf_to_text(raw: bytes) -> str:
    """Extract text from PDF bytes; returns '' if it can't be parsed."""
    try:
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception:
        return ""


async def _upload_to_text(upload: UploadFile) -> str:
    """Read an uploaded resume file and return its text (handles PDF and plain text)."""
    raw = await upload.read()
    filename = (upload.filename or "").lower()
    content_type = (upload.content_type or "").lower()
    if filename.endswith(".pdf") or "pdf" in content_type:
        return _pdf_to_text(raw)
    return raw.decode("utf-8", errors="ignore").strip()


async def _extract_cv_and_role(request: Request) -> tuple[str, str]:
    """Pull (cv_text, role) from either a JSON body or a multipart/form-data upload."""
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("application/json"):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid JSON body.")
        if not isinstance(body, dict):
            raise HTTPException(status_code=422, detail="JSON body must be an object.")
        return str(body.get("cv") or "").strip(), str(body.get("role") or "").strip()

    # Otherwise treat it as form data (what the PDF-upload frontend sends).
    form = await request.form()
    items = form.multi_items()
    text_fields = [(k, v) for k, v in items if isinstance(v, str)]
    file_fields = [(k, v) for k, v in items if isinstance(v, UploadFile)]

    # Role: a role-named text field, else the only text field if there's just one.
    role = ""
    for key, value in text_fields:
        if key.lower() in _ROLE_FIELDS and value.strip():
            role = value.strip()
            break
    if not role and len(text_fields) == 1 and text_fields[0][1].strip():
        role = text_fields[0][1].strip()

    # CV: prefer an uploaded file, else a cv-named text field.
    cv = ""
    for _key, upload in file_fields:
        cv = await _upload_to_text(upload)
        if cv:
            break
    if not cv:
        for key, value in text_fields:
            if key.lower() in _CV_TEXT_FIELDS and value.strip():
                cv = value.strip()
                break

    return cv, role


def _build_ranking_prompt(cv: str, role: str, batch: list[tuple[int, dict]]) -> str:
    """Build the Gemini prompt for one batch of (index, job) pairs."""
    listing = "\n".join(
        f"{idx}. {job.get('title')} at {job.get('company')} "
        f"(location: {job.get('location') or 'N/A'})"
        for idx, job in batch
    )
    return (
        "You are a technical recruiter. Score how well each job below fits the "
        "candidate, based on their CV and the role they want.\n\n"
        f"TARGET ROLE THE CANDIDATE WANTS:\n{role}\n\n"
        f"CANDIDATE CV:\n{cv}\n\n"
        "JOBS (each line is `index. title at company (location)`):\n"
        f"{listing}\n\n"
        "For every job, return an object with:\n"
        "- index: the job's index exactly as given above\n"
        "- score: an integer 0-100 for fit (100 = perfect fit, 0 = no fit), "
        "weighing the desired role, CV skills/experience, and seniority\n"
        "- reason: one short sentence (max ~20 words) explaining the score\n"
        "Return a result for every job index in the list."
    )


RANK_RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "index": {"type": "INTEGER"},
            "score": {"type": "INTEGER"},
            "reason": {"type": "STRING"},
        },
        "required": ["index", "score", "reason"],
    },
}


async def _score_batch(
    client: httpx.AsyncClient,
    api_key: str,
    cv: str,
    role: str,
    batch: list[tuple[int, dict]],
) -> dict[int, dict]:
    """Ask Gemini to score one batch of jobs; return {index: {score, reason}}."""
    payload = {
        "contents": [
            {"parts": [{"text": _build_ranking_prompt(cv, role, batch)}]}
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RANK_RESPONSE_SCHEMA,
            "temperature": 0.2,
        },
    }
    response = await client.post(
        GEMINI_URL.format(model=GEMINI_MODEL),
        headers={"x-goog-api-key": api_key},
        json=payload,
    )
    response.raise_for_status()
    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    scored = json.loads(text)

    valid_indices = {idx for idx, _ in batch}
    out: dict[int, dict] = {}
    for item in scored:
        idx = item.get("index")
        if idx not in valid_indices:
            continue
        score = max(0, min(100, int(item.get("score", 0))))
        out[idx] = {"score": score, "reason": item.get("reason", "")}
    return out


@app.post("/rank")
async def rank_jobs(request: Request) -> dict:
    """Rank all jobs by how well they fit the given CV and desired role via Gemini.

    Accepts either a JSON body ``{"cv": "...", "role": "..."}`` or a
    ``multipart/form-data`` upload with a resume file (PDF or text) plus a
    ``role`` field, so browser file uploads work directly.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    cv, role = await _extract_cv_and_role(request)
    if not cv:
        raise HTTPException(
            status_code=422,
            detail="Missing CV: provide 'cv' text (JSON) or upload a resume file "
            "(form-data). If you uploaded a PDF, it may be scanned/image-only with "
            "no extractable text.",
        )
    if not role:
        raise HTTPException(status_code=422, detail="Missing 'role'.")

    async with httpx.AsyncClient(timeout=60.0) as client:
        jobs = await fetch_all_jobs(client)

        indexed = list(enumerate(jobs))
        batches = [
            indexed[i : i + RANK_BATCH_SIZE]
            for i in range(0, len(indexed), RANK_BATCH_SIZE)
        ]

        semaphore = asyncio.Semaphore(RANK_CONCURRENCY)

        async def run(batch: list[tuple[int, dict]]) -> dict[int, dict]:
            async with semaphore:
                return await _score_batch(client, api_key, cv, role, batch)

        batch_results = await asyncio.gather(
            *(run(batch) for batch in batches),
            return_exceptions=True,
        )

    scores: dict[int, dict] = {}
    for result in batch_results:
        if isinstance(result, Exception):
            # Degrade gracefully: a failed batch leaves its jobs unscored below.
            continue
        scores.update(result)

    ranked: list[dict] = []
    for idx, job in indexed:
        scored = scores.get(idx)
        ranked.append(
            {
                "title": job.get("title"),
                "company": job.get("company"),
                "location": job.get("location"),
                "apply_url": job.get("apply_url"),
                "score": scored["score"] if scored else 0,
                "reason": (
                    scored["reason"]
                    if scored
                    else "Could not be scored (ranking service error)."
                ),
            }
        )

    ranked.sort(key=lambda job: job["score"], reverse=True)
    return {"count": len(ranked), "jobs": ranked}
