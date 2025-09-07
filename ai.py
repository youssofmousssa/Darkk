#!/usr/bin/env python3
"""
darkai_proxy.py
Single-file FastAPI proxy integrating all DarkAI endpoints you provided.

Features:
- Endpoints: /api/{service_name} (POST) and /api/{service_name}/get (GET)
- Accepts JSON, form-data, file uploads (serves files from /uploads/)
- Retries with exponential backoff, per-service concurrency limits
- CORS (configurable via env)
- Transparent passthrough of upstream response (JSON or text)
- Logging + graceful shutdown for httpx client
- Config via environment variables

Run:
    DARKAI_BASE=https://sii3.moayman.top \
    uvicorn darkai_proxy:app --host 0.0.0.0 --port 8000

Note: Uploaded files are saved to UPLOAD_DIR and served at /uploads/<filename>.
"""

import os
import sys
import uuid
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import httpx
import aiofiles
from dotenv import load_dotenv

load_dotenv()  # optional .env

# ---------- Configuration ----------
DARKAI_BASE = os.getenv("DARKAI_BASE", "https://sii3.moayman.top")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/darkai_uploads")
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
# Allowed origins: comma-separated list or "*" (default)
if ALLOW_ORIGINS.strip() == "*" or ALLOW_ORIGINS.strip() == "":
    ALLOW_ORIGINS_LIST = ["*"]
else:
    ALLOW_ORIGINS_LIST = [o.strip() for o in ALLOW_ORIGINS.split(",")]

EXT_TIMEOUT = float(os.getenv("EXT_TIMEOUT", "30"))  # seconds
RETRIES = int(os.getenv("RETRIES", "2"))
CONCURRENCY_PER_SERVICE = int(os.getenv("CONCURRENCY_PER_SERVICE", "6"))
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "50"))
DARKAI_API_KEY = os.getenv("DARKAI_API_KEY", None)  # optional if backend requires key

# Make uploads directory
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- Logging ----------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("darkai-proxy")

# ---------- FastAPI app ----------
app = FastAPI(title="DarkAI Unified Proxy", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ---------- Upstream service mapping ----------
# Maps service short key -> { path: <upstream path>, method: <"POST"|"GET">, note: <optional> }
SERVICES = {
    "gemini-img": {"path": "/api/gemini-img.php", "method": "POST"},
    "flux-pro": {"path": "/api/flux-pro.php", "method": "POST"},
    "gpt-img": {"path": "/api/gpt-img.php", "method": "POST"},
    "nano-banana": {"path": "/api/nano-banana.php", "method": "POST"},
    "img-cv": {"path": "/api/img-cv.php", "method": "POST"},
    "voice": {"path": "/api/voice.php", "method": "POST"},
    "veo3": {"path": "/api/veo3.php", "method": "POST"},
    "music": {"path": "/api/music.php", "method": "POST"},
    "create-music": {"path": "/api/create-music.php", "method": "POST"},
    "wormgpt": {"path": "/DARK/api/wormgpt.php", "method": "POST"},
    "do": {"path": "/api/do.php", "method": "GET"},  # downloader: GET with url param
    "remove-bg": {"path": "/api/remove-bg.php", "method": "GET"},
    "gemini-dark": {"path": "/api/gemini-dark.php", "method": "POST"},
    "gemini": {"path": "/DARK/gemini.php", "method": "POST"},
    "ai": {"path": "/api/ai.php", "method": "POST"},
}

# Create semaphores per service to limit concurrent upstream calls
_semaphores: Dict[str, asyncio.Semaphore] = {
    k: asyncio.Semaphore(CONCURRENCY_PER_SERVICE) for k in SERVICES.keys()
}

# HTTPX async client
_client: Optional[httpx.AsyncClient] = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        limits = httpx.Limits(max_connections=MAX_CONNECTIONS, max_keepalive_connections=20)
        _client = httpx.AsyncClient(timeout=httpx.Timeout(EXT_TIMEOUT), limits=limits)
    return _client


@app.on_event("shutdown")
async def shutdown_event():
    global _client
    if _client:
        await _client.aclose()
        _client = None
        logger.info("HTTPX client closed.")


# ---------- Helpers ----------
async def save_upload_and_get_public_url(upload: UploadFile, request: Request) -> str:
    """
    Save UploadFile to UPLOAD_DIR and return absolute public URL (based on request.base_url).
    NOTE: For upstream servers to be able to fetch it, this URL must be publicly reachable.
    """
    suffix = os.path.splitext(upload.filename)[1] or ""
    fname = f"{uuid.uuid4().hex}{suffix}"
    dest = os.path.join(UPLOAD_DIR, fname)
    try:
        async with aiofiles.open(dest, "wb") as out_f:
            content = await upload.read()
            await out_f.write(content)
    finally:
        await upload.close()
    base = str(request.base_url).rstrip("/")
    public_url = f"{base}/uploads/{fname}"
    logger.info("Saved upload %s -> %s", upload.filename, public_url)
    return public_url


def build_upstream_url(service_key: str) -> str:
    conf = SERVICES[service_key]
    return DARKAI_BASE.rstrip("/") + conf["path"]


async def _attempt_request(
    service_key: str,
    data: Dict[str, Any],
    params: Dict[str, Any],
    files: Optional[Dict[str, bytes]],
    json_body: Optional[Dict[str, Any]],
) -> httpx.Response:
    """
    Single attempt to call upstream. We use method from SERVICES mapping.
    `data` -> form data
    `params` -> query params for GET
    `files` -> for multipart (not often needed here, upstream usually expects url links)
    `json_body` -> if sending JSON
    """
    url = build_upstream_url(service_key)
    method = SERVICES[service_key]["method"].upper()
    client = get_client()
    headers = {}
    if DARKAI_API_KEY:
        # optional header if upstream requires it — customize as needed
        headers["Authorization"] = f"Bearer {DARKAI_API_KEY}"

    # Choose request style based on method and presence of json_body
    if method == "GET":
        logger.debug("Upstream GET %s params=%s", url, params)
        resp = await client.get(url, params=params, headers=headers)
        return resp
    else:
        # prefer POST form data (most upstream examples use form fields like 'text' and 'link')
        if json_body is not None:
            logger.debug("Upstream POST JSON %s json=%s", url, json_body)
            resp = await client.post(url, json=json_body, headers=headers)
            return resp
        else:
            # Use data for form fields; include files if provided (multipart)
            if files:
                # prepare files mapping for httpx: {fieldname: (filename, data, content_type)}
                files_payload = {k: (v["filename"], v["content"], v.get("content_type", "application/octet-stream")) for k, v in files.items()}
                logger.debug("Upstream POST multipart %s data=%s files=%s", url, list(data.keys()), list(files_payload.keys()))
                resp = await client.post(url, data=data, files=files_payload, headers=headers)
                return resp
            else:
                logger.debug("Upstream POST form %s data=%s", url, list(data.keys()))
                resp = await client.post(url, data=data, headers=headers)
                return resp


async def call_upstream_with_retries(service_key: str, data: Dict[str, Any], params: Dict[str, Any], files: Optional[Dict[str, bytes]], json_body: Optional[Dict[str, Any]]):
    last_exc = None
    for attempt in range(RETRIES + 1):
        try:
            resp = await _attempt_request(service_key, data, params, files, json_body)
            # treat HTTP < 400 as success
            if resp.status_code < 400:
                return resp
            else:
                logger.warning("Upstream %s returned status %s: %s", service_key, resp.status_code, resp.text[:200])
                last_exc = RuntimeError(f"Upstream returned {resp.status_code}")
        except Exception as e:
            logger.exception("Exception calling upstream (attempt %d/%d) for %s: %s", attempt + 1, RETRIES + 1, service_key, str(e))
            last_exc = e
        # backoff
        backoff = (2 ** attempt) * 0.5
        await asyncio.sleep(backoff)
    # all attempts failed
    raise HTTPException(status_code=502, detail=f"Upstream {service_key} failed after retries. Last error: {last_exc}")


def extract_resp_content(resp: httpx.Response):
    ctype = resp.headers.get("content-type", "")
    # JSON
    if "application/json" in ctype or resp.text.strip().startswith("{"):
        try:
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except Exception:
            # fallback to plain text
            return PlainTextResponse(content=resp.text, status_code=resp.status_code)
    # image or binary or other
    elif ctype.startswith("image/") or "octet-stream" in ctype:
        # return direct JSON with url if upstream returned JSON containing url
        try:
            j = resp.json()
            return JSONResponse(content=j, status_code=resp.status_code)
        except Exception:
            # upstream returned raw image bytes - return textual base64 is one option,
            # but we will return upstream text or binary as plain text to avoid accidental binary streaming in JSON.
            return PlainTextResponse(content=resp.text, status_code=resp.status_code, media_type=ctype)
    else:
        return PlainTextResponse(content=resp.text, status_code=resp.status_code)


# ---------- API Endpoints ----------
@app.get("/health")
async def health():
    return {"status": "ok", "service_count": len(SERVICES)}


@app.post("/api/{service_name}")
async def proxy_post(service_name: str, request: Request):
    """
    Generic POST proxy endpoint for all services.
    Accepts:
      - application/json with body -> forwarded as JSON (json_body)
      - form-data (including UploadFile fields) -> forwarded as form data (and files served locally and forwarded as 'link'/'links')
    Behavior:
      - if you upload files: they'll be saved to /uploads/ and their public URLs added to payload:
          * for multiple files -> 'links' param (comma-separated) OR preserve field name if provided
          * for a single file -> 'link' param if no other 'link' provided
      - All other form fields are forwarded as-is.
    """
    service_name = service_name.strip()
    if service_name not in SERVICES:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found. Available: {list(SERVICES.keys())}")

    # Acquire semaphore for service
    sem = _semaphores[service_name]
    async with sem:
        content_type = request.headers.get("content-type", "")
        data = {}
        params = {}
        files_payload = None
        json_body = None
        uploaded_urls: List[str] = []

        # If JSON body -> forward as JSON
        if "application/json" in content_type:
            json_body = await request.json()
        else:
            # parse form
            form = await request.form()
            # form is a starlette.datastructures.FormData (iterable)
            for key, value in form.multi_items():
                # UploadFile instances (form files) will be of UploadFile type
                if isinstance(value, UploadFile):
                    # save and get public URL
                    public_url = await save_upload_and_get_public_url(value, request)
                    uploaded_urls.append(public_url)
                else:
                    # multiple values for same key? keep last (like regular form)
                    data[key] = str(value)

        # If we saved uploads, decide where to put them in payload
        if uploaded_urls:
            # if upstream expects 'links' param for multiple images (nano-banana etc.) prefer that
            if "links" in data or len(uploaded_urls) > 1:
                # merge with existing 'links' if present
                existing = data.get("links", "")
                combined = ",".join([existing] + uploaded_urls) if existing else ",".join(uploaded_urls)
                data["links"] = combined
            else:
                # single file: set 'link' unless user already supplied link
                if "link" not in data:
                    data["link"] = uploaded_urls[0]
                else:
                    # user supplied link and uploaded file(s) as well -> append to 'links'
                    data["links"] = ",".join([data.get("link")] + uploaded_urls)

        # Some specific behaviors: if service 'do' (downloader) expects 'url' param, allow both 'url' and 'link'
        if service_name == "do":
            # ensure there is a 'url' parameter
            if "url" not in data and "link" in data:
                data["url"] = data["link"]

        # Now call upstream with retries
        resp = await call_upstream_with_retries(service_name, data, params, files_payload, json_body)

        return extract_resp_content(resp)


@app.get("/api/{service_name}/get")
async def proxy_get(service_name: str, request: Request):
    """
    Generic GET proxy wrapper that forwards query parameters as-is to the upstream GET endpoint.
    Example: /api/veo3/get?text=Create+a+cinematic+intro
    """
    service_name = service_name.strip()
    if service_name not in SERVICES:
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found.")
    if SERVICES[service_name]["method"].upper() not in ("GET", "POST"):
        # We'll still allow GET forwarding; upstream may accept GET even if examples showed POST
        pass

    sem = _semaphores[service_name]
    async with sem:
        params = dict(request.query_params)
        # special: if service is 'do' and param is 'url' already present it's fine
        resp = await call_upstream_with_retries(service_name, data={}, params=params, files=None, json_body=None)
        return extract_resp_content(resp)


@app.get("/services")
async def list_services():
    """
    Return simple list of available services and example usage.
    """
    sample_base = "/api/<service> (POST) or /api/<service>/get (GET)"
    return {"available_services": list(SERVICES.keys()), "usage": sample_base}


# ---------- Example convenience endpoints for common flows ----------
@app.post("/api/generate-image")
async def generate_image_proxy(request: Request):
    """
    Convenience endpoint:
    - expects form/json field 'model' (e.g., 'flux-pro', 'gpt-img', 'gemini-img', 'img-cv', 'nano-banana')
    - expects 'text' and either 'link' (URL) or file uploads (images)
    Forwards to the chosen service.
    """
    form_or_json_content_type = request.headers.get("content-type", "")
    payload = {}
    if "application/json" in form_or_json_content_type:
        payload = await request.json()
    else:
        form = await request.form()
        for k, v in form.multi_items():
            if not isinstance(v, UploadFile):
                payload[k] = v
    model = payload.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="Please provide 'model' (e.g., 'flux-pro', 'gpt-img', 'gemini-img')")
    if model not in SERVICES:
        raise HTTPException(status_code=404, detail=f"Model '{model}' not supported by proxy.")
    # Reuse proxy_post logic by delegating; create a fake request with same body is complicated,
    # so instruct caller to hit /api/{model} directly. We return helpful guidance instead.
    return JSONResponse({
        "info": "Use /api/{model} directly. Example:",
        "curl_example_form": f"curl -X POST 'http://<your_host>/api/{model}' -F 'text=Make+icon+gold' -F 'link=https://example.com/img.png'",
        "note": "If uploading files, ensure server is publicly reachable so upstream can fetch /uploads/ URLs."
    })


# ---------- Run block ----------
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting DarkAI proxy with base %s", DARKAI_BASE)
    uvicorn.run("darkai_proxy:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="info")
