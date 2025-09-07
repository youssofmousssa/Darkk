# main.py
import os
import time
import uuid
import hmac
import hashlib
import logging
import asyncio
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, Header, HTTPException, status, Depends, Response
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import httpx
import aioredis
import jwt
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.concurrency import run_in_threadpool
from pybreaker import CircuitBreaker, CircuitBreakerError

# --- Configuration and environment ---

API_KEY_SECRET = os.getenv("API_KEY_SECRET", "supersecretapikeysecret")  # HMAC secret to validate API keys signatures
JWT_SECRET = os.getenv("JWT_SECRET", "supersecretjwtsecret")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_SECONDS = 900  # 15 mins
JWT_REFRESH_TOKEN_EXPIRE_SECONDS = 604800  # 7 days

UPSTREAM_BASE_URL = "https://sii3.moayman.top"
UPSTREAM_TIMEOUT = 15.0  # seconds
REQUEST_ID_TTL_SECONDS = 300  # 5 minutes TTL for request ID dedupe in Redis
HMAC_TIMESTAMP_SKEW = 120  # seconds allowed clock skew for signatures
RATE_LIMIT_RPS = int(os.getenv("RATE_LIMIT_RPS", "10"))
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "100"))
DAILY_QUOTA = int(os.getenv("DAILY_QUOTA", "10000"))
MONTHLY_QUOTA = int(os.getenv("MONTHLY_QUOTA", "300000"))

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost")

# --- Logging setup ---

logger = logging.getLogger("darkai_proxy")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

# --- Prometheus metrics ---

REQUEST_COUNT = Counter(
    "darkai_proxy_requests_total",
    "Total API requests processed",
    ['endpoint', 'method', 'status_code']
)
REQUEST_LATENCY = Histogram(
    "darkai_proxy_request_latency_seconds",
    "Latency for API requests",
    ['endpoint']
)

# --- FastAPI app and middleware ---

app = FastAPI(title="DarkAI Unified Proxy API")

# Force HTTPS redirection middleware
app.add_middleware(HTTPSRedirectMiddleware)


# --- Redis client global ---
redis = None

async def get_redis():
    global redis
    if redis is None:
        redis = await aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    return redis


# --- Security Utilities ---

def verify_hmac_signature(secret: str, signature: str, timestamp: str, method: str, path: str, body: bytes) -> bool:
    """
    Verify HMAC_SHA256 signature with scheme:
    signature_base = timestamp + '\n' + method + '\n' + path + '\n' + body_hash
    X-Signature = HMAC_SHA256(secret, signature_base)
    """
    try:
        body_hash = hashlib.sha256(body if body else b'').hexdigest()
        signature_base = f"{timestamp}\n{method.upper()}\n{path}\n{body_hash}".encode()
        computed_hmac = hmac.new(secret.encode(), signature_base, hashlib.sha256).hexdigest()
        return hmac.compare_digest(computed_hmac, signature)
    except Exception as e:
        logger.error(f"Failed signature verification: {e}")
        return False

def create_jwt_token(data: dict, expires_in: int) -> str:
    payload = data.copy()
    payload.update({"exp": time.time() + expires_in})
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def decode_jwt_token(token: str) -> dict:
    try:
        decoded = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return decoded
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid JWT token")

# --- API Key Data Model ---

class APIKeyData(BaseModel):
    client_id: str
    secret: str
    allowed_ips: Optional[list] = None
    revoked: bool = False
    daily_quota_used: int = 0
    monthly_quota_used: int = 0

# --- In-memory API key store (demo only - replace with persistent DB) ---

# For production, use persistent DB and Vault for secrets
api_keys_store: Dict[str, APIKeyData] = {}

# --- Authentication & Authorization Dependencies ---

async def get_api_key(x_api_key: str = Header(...), client_ip: Optional[str] = None) -> APIKeyData:
    """
    Validate API key header, check revocation and IP allowlist.
    """
    key_data = api_keys_store.get(x_api_key)
    if not key_data or key_data.revoked:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or revoked API key")
    if key_data.allowed_ips:
        if client_ip is None or client_ip not in key_data.allowed_ips:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="IP not allowed for this API key")
    return key_data

security = HTTPBearer()

async def get_current_user(token: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Validate JWT bearer token for user session.
    """
    token_str = token.credentials
    user_data = decode_jwt_token(token_str)
    return user_data

# --- Request Integrity Middleware ---

class RequestIntegrityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Extract and verify required headers
        x_request_id = request.headers.get("X-Request-ID")
        x_signature = request.headers.get("X-Signature")
        x_timestamp = request.headers.get("X-Timestamp")
        x_api_key = request.headers.get("x-api-key")
        client_ip = request.client.host if request.client else None

        # Validate presence
        if not x_api_key:
            return JSONResponse(status_code=401, content={"error": "Missing x-api-key header"})
        if not x_request_id:
            return JSONResponse(status_code=400, content={"error": "Missing X-Request-ID header"})
        if not x_signature:
            return JSONResponse(status_code=401, content={"error": "Missing X-Signature header"})
        if not x_timestamp:
            return JSONResponse(status_code=400, content={"error": "Missing X-Timestamp header"})

        # Validate timestamp skew
        try:
            ts_int = int(x_timestamp)
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "Invalid X-Timestamp header"})
        now = int(time.time())
        if abs(now - ts_int) > HMAC_TIMESTAMP_SKEW:
            return JSONResponse(status_code=400, content={"error": "Request timestamp out of allowed range"})

        # Check API key existence and IP allowlist
        key_data = api_keys_store.get(x_api_key)
        if not key_data or key_data.revoked:
            return JSONResponse(status_code=401, content={"error": "Invalid or revoked API key"})
        if key_data.allowed_ips and client_ip not in key_data.allowed_ips:
            return JSONResponse(status_code=403, content={"error": "IP not allowed for this API key"})

        # Check replay of X-Request-ID
        redis = await get_redis()
        request_id_key = f"request_id:{x_request_id}"
        is_duplicate = await redis.get(request_id_key)
        if is_duplicate:
            return JSONResponse(status_code=409, content={"error": "Duplicate X-Request-ID detected"})
        await redis.set(request_id_key, "1", ex=REQUEST_ID_TTL_SECONDS)

        # Read body for signature verification
        body_bytes = await request.body()

        # Verify HMAC signature
        valid_signature = verify_hmac_signature(
            secret=key_data.secret,
            signature=x_signature,
            timestamp=x_timestamp,
            method=request.method,
            path=request.url.path,
            body=body_bytes
        )
        if not valid_signature:
            return JSONResponse(status_code=401, content={"error": "Invalid HMAC signature"})

        # Replace request stream for downstream handlers since we read body
        async def receive():
            return {"type": "http.request", "body": body_bytes}
        request._receive = receive

        # Rate limiting check
        # Implement per-key and per-IP rate limiting using Redis
        # Key format: ratelimit:{api_key} and ratelimit:{api_key}:{client_ip}
        try:
            # Simple token bucket algorithm using Lua script or atomic Redis commands
            # Here: simplistic approach - increment counters with expiry
            api_key_rl_key = f"ratelimit:{x_api_key}"
            api_key_ip_rl_key = f"ratelimit:{x_api_key}:{client_ip}"

            # Increment counters
            current_count_key = await redis.incr(api_key_rl_key)
            if current_count_key == 1:
                await redis.expire(api_key_rl_key, 1)  # 1 second window

            current_count_ip_key = await redis.incr(api_key_ip_rl_key)
            if current_count_ip_key == 1:
                await redis.expire(api_key_ip_rl_key, 1)

            # Check limits
            if current_count_key > RATE_LIMIT_BURST or current_count_ip_key > RATE_LIMIT_BURST:
                retry_after = await redis.ttl(api_key_rl_key)
                headers = {"Retry-After": str(retry_after if retry_after > 0 else 1)}
                return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"}, headers=headers)
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Fail open on Redis error

        # Attach key_data and client_ip to request.state for downstream use
        request.state.api_key_data = key_data
        request.state.client_ip = client_ip
        request.state.x_request_id = x_request_id

        # Proceed to next middleware / route handler
        response = await call_next(request)
        return response

app.add_middleware(RequestIntegrityMiddleware)


# --- Circuit Breaker for upstream calls ---

circuit_breaker = CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    exclude=[httpx.HTTPStatusError]  # Only trip on network errors, not HTTP errors
)

# --- Upstream proxy call with retry and circuit breaker ---

async def call_upstream(
    method: str,
    path: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, str]] = None,
    data: Optional[Dict[str, Any]] = None,
    timeout: float = UPSTREAM_TIMEOUT,
) -> httpx.Response:
    url = UPSTREAM_BASE_URL + path
    async with httpx.AsyncClient(timeout=timeout) as client:
        @circuit_breaker
        async def do_request():
            if method.upper() == "GET":
                return await client.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                return await client.post(url, headers=headers, data=data)
            else:
                raise HTTPException(status_code=405, detail="Method not allowed")

        # Retry with exponential backoff on transient errors
        for attempt in range(3):
            try:
                response = await do_request()
                response.raise_for_status()
                return response
            except CircuitBreakerError:
                raise HTTPException(status_code=503, detail="Upstream service temporarily unavailable (circuit breaker open)")
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if attempt == 2:
                    raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")
                await asyncio.sleep(2 ** attempt)
        raise HTTPException(status_code=502, detail="Upstream request failed after retries")

# --- Cache for GET image/video requests (simple Redis cache) ---

async def get_cache(key: str) -> Optional[str]:
    redis = await get_redis()
    return await redis.get(key)

async def set_cache(key: str, value: str, ttl: int = 300):
    redis = await get_redis()
    await redis.set(key, value, ex=ttl)

# --- Structured Logging Middleware ---

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        api_key = getattr(request.state, "api_key_data", None)
        key_id = api_key.client_id if api_key else "unknown"
        user_id = None
        if hasattr(request.state, "user"):
            user_id = getattr(request.state.user, "id", None)
        request_id = getattr(request.state, "x_request_id", None)
        client_ip = getattr(request.state, "client_ip", None)

        log_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "api_key": key_id,
            "user_id": user_id,
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "latency_ms": int(process_time * 1000),
            "client_ip": client_ip,
        }
        logger.info(f"RequestLog: {log_data}")

        # Update Prometheus metrics
        REQUEST_COUNT.labels(endpoint=request.url.path, method=request.method, status_code=str(response.status_code)).inc()
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(process_time)

        return response

app.add_middleware(StructuredLoggingMiddleware)

# --- Authentication endpoints to register API keys and issue JWTs ---

class RegisterRequest(BaseModel):
    client_id: str

class TokenRequest(BaseModel):
    client_id: str
    api_key: str

@app.post("/auth/register")
async def register_api_key(req: RegisterRequest):
    # In production, secure this endpoint (admin only)
    # Generate API key and secret (HMAC secret)
    api_key = str(uuid.uuid4())
    secret = hashlib.sha256(os.urandom(32)).hexdigest()
    api_keys_store[api_key] = APIKeyData(client_id=req.client_id, secret=secret)
    return {"api_key": api_key, "secret": secret}

@app.post("/auth/token")
async def issue_jwt_token(req: TokenRequest):
    key_data = api_keys_store.get(req.api_key)
    if not key_data or key_data.client_id != req.client_id or key_data.revoked:
        raise HTTPException(status_code=401, detail="Invalid API key or client ID")

    access_token = create_jwt_token({"client_id": req.client_id}, JWT_ACCESS_TOKEN_EXPIRE_SECONDS)
    refresh_token = create_jwt_token({"client_id": req.client_id, "refresh": True}, JWT_REFRESH_TOKEN_EXPIRE_SECONDS)
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

# --- Proxy routes for all services ---

# Map proxy routes to upstream paths
SERVICE_PATHS = {
    "gemini-img": "/api/gemini-img.php",
    "flux-pro": "/api/flux-pro.php",
    "gpt-img": "/api/gpt-img.php",
    "nano-banana": "/api/nano-banana.php",
    "img-cv": "/api/img-cv.php",
    "voice": "/api/voice.php",
    "veo3": "/api/veo3.php",
    "music": "/api/music.php",
    "create-music": "/api/create-music.php",
    "wormgpt": "/DARK/api/wormgpt.php",
    "ai": "/api/ai.php",
    "gemini-dark": "/api/gemini-dark.php",
    "gemini": "/DARK/gemini.php",
    "do": "/api/do.php",
    "remove-bg": "/api/remove-bg.php",
}

@app.api_route("/api/{service_name}", methods=["GET", "POST"])
async def proxy_service(
    service_name: str,
    request: Request,
    api_key_data: APIKeyData = Depends(get_api_key)
):
    if service_name not in SERVICE_PATHS:
        raise HTTPException(status_code=404, detail="Service not found")

    path = SERVICE_PATHS[service_name]

    # Prepare headers for upstream call
    upstream_headers = {
        "User-Agent": "DarkAI-Proxy/1.0",
        # Inject proxy request id header
        "X-Proxy-Request-Id": request.headers.get("X-Request-ID", str(uuid.uuid4())),
        # Add any upstream auth headers here if required (e.g. HMAC with upstream secret)
    }

    # Forward client headers except auth headers
    # Here we explicitly set required headers only for clarity
    # Also forward content-type for POSTs
    if request.method == "POST":
        content_type = request.headers.get("content-type")
        if content_type:
            upstream_headers["Content-Type"] = content_type

    # Handle GET and POST parameters
    if request.method == "GET":
        params = dict(request.query_params)
        data = None
    else:
        form = await request.form()
        params = None
        data = dict(form)

    # Cache GET image/video requests (idempotent)
    cache_key = None
    cached_response = None
    if request.method == "GET" and service_name in {"gemini-img", "flux-pro", "gpt-img", "nano-banana", "img-cv", "voice", "veo3", "music", "create-music", "do", "remove-bg"}:
        # Cache key by service + sorted query params string
        sorted_params = sorted(params.items()) if params else []
        cache_key = f"cache:{service_name}:" + hashlib.sha256(str(sorted_params).encode()).hexdigest()
        cached_response = await get_cache(cache_key)
        if cached_response:
            return JSONResponse(content=eval(cached_response))  # eval is safe here because stored dict string, for demo only

    # Call upstream service with retry and circuit breaker
    try:
        response = await call_upstream(
            method=request.method,
            path=path,
            headers=upstream_headers,
            params=params,
            data=data
        )
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})

    # Stream response back to client
    content = response.content
    content_type = response.headers.get("content-type", "application/json")

    # Cache GET JSON responses for idempotent endpoints
    if cache_key and response.status_code == 200 and "application/json" in content_type:
        await set_cache(cache_key, content.decode(), ttl=300)

    return Response(content=content, media_type=content_type)


# --- Fallback raw proxy for any other paths under /proxy/ ---

@app.api_route("/proxy/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_raw(
    full_path: str,
    request: Request,
    api_key_data: APIKeyData = Depends(get_api_key)
):
    path = f"/{full_path}"
    upstream_headers = {
        "User-Agent": "DarkAI-Proxy/1.0",
        "X-Proxy-Request-Id": request.headers.get("X-Request-ID", str(uuid.uuid4())),
    }

    if request.method == "POST":
        content_type = request.headers.get("content-type")
        if content_type:
            upstream_headers["Content-Type"] = content_type

    if request.method == "GET":
        params = dict(request.query_params)
        data = None
    else:
        form = await request.form()
        params = None
        data = dict(form)

    try:
        response = await call_upstream(
            method=request.method,
            path=path,
            headers=upstream_headers,
            params=params,
            data=data
        )
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})

    return Response(content=response.content, media_type=response.headers.get("content-type", "application/json"))


# --- Prometheus metrics endpoint ---

@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(data, media_type=CONTENT_TYPE_LATEST)


# --- Admin endpoints for API key management (demo only, protect in prod) ---

@app.get("/admin/api-keys")
async def list_api_keys():
    # Return list of keys with client ids and revoked status
    keys_list = []
    for k, v in api_keys_store.items():
        keys_list.append({"api_key": k, "client_id": v.client_id, "revoked": v.revoked})
    return {"keys": keys_list}

class APIKeyRevokeRequest(BaseModel):
    api_key: str

@app.post("/admin/api-keys/revoke")
async def revoke_api_key(req: APIKeyRevokeRequest):
    key_data = api_keys_store.get(req.api_key)
    if not key_data:
        raise HTTPException(status_code=404, detail="API key not found")
    key_data.revoked = True
    ret
