import base64
import json
import os
import time
from collections import defaultdict, deque

from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS")
ALLOWED_ORIGINS = (
    [origin.strip() for origin in FRONTEND_ORIGINS.split(",") if origin.strip()]
    if FRONTEND_ORIGINS
    else DEFAULT_ALLOWED_ORIGINS
)

MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024
RATE_LIMIT_WINDOW_SECONDS = 10 * 60
RATE_LIMIT_REQUEST_COUNT = 5

request_history = defaultdict(deque)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


ATS_SYSTEM_PROMPT = """
You are an expert ATS resume evaluator and hiring assistant.

Your job is to compare the candidate CV with the provided job description and return only valid JSON.

Rules:
1. Return only JSON. No markdown. No explanation outside JSON.
2. Keep feedback practical, concise, and ATS-focused.
3. Score should be an integer from 0 to 100.
4. scoreLabel should be a short phrase like "Low match", "Moderate match", or "Strong match".
5. missingKeywords, strengths, weakAreas, and suggestedImprovements must each contain 3 to 6 short bullet-style strings.
6. rewrittenSummary must be a short professional summary tailored to the job.

Required JSON shape:
{
  "score": 0,
  "scoreLabel": "",
  "missingKeywords": [],
  "strengths": [],
  "weakAreas": [],
  "suggestedImprovements": [],
  "rewrittenSummary": ""
}
""".strip()


def normalize_text_upload(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return file_bytes.decode("latin-1")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Unable to read uploaded text file.")


def enforce_allowed_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin and origin not in ALLOWED_ORIGINS:
        raise HTTPException(status_code=403, detail="Origin is not allowed.")


def enforce_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    timestamps = request_history[client_ip]

    while timestamps and now - timestamps[0] > RATE_LIMIT_WINDOW_SECONDS:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_REQUEST_COUNT:
        raise HTTPException(
            status_code=429,
            detail="Rate limit reached. Please wait before analyzing another resume.",
        )

    timestamps.append(now)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/ats-analyze")
async def ats_analyze(
    request: Request,
    cv_file: UploadFile = File(...),
    job_description: str = Form(...),
):
    enforce_allowed_origin(request)
    enforce_rate_limit(request)

    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description is required.")

    if not cv_file.filename:
        raise HTTPException(status_code=400, detail="Uploaded CV file must have a file name.")

    file_bytes = await cv_file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded CV file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="CV file is too large. Max size is 2 MB.")

    content_blocks = []
    filename = cv_file.filename.lower()

    if filename.endswith(".pdf"):
        content_blocks.append(
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.b64encode(file_bytes).decode("utf-8"),
                },
            }
        )
        content_blocks.append(
            {
                "type": "text",
                "text": f"Analyze the uploaded CV against this job description:\n\n{job_description}",
            }
        )
    elif filename.endswith(".txt"):
        cv_text = normalize_text_upload(file_bytes)
        content_blocks.append(
            {
                "type": "text",
                "text": f"Candidate CV:\n{cv_text}\n\nJob Description:\n{job_description}",
            }
        )
    else:
        raise HTTPException(status_code=400, detail="Version 1 supports only .pdf and .txt CV files.")

    try:
        response = client.messages.create(
            model="claude-sonnet-4-0",
            max_tokens=1200,
            system=ATS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content_blocks}],
            temperature=0,
        )

        raw_text = response.content[0].text.strip()
        return json.loads(raw_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model response was not valid JSON.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
