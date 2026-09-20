import os
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
UPSTREAM = os.getenv("BRAIN_UPSTREAM_URL", "").rstrip("/")
RELAY_API_KEY = os.getenv("RELAY_API_KEY", "")
BRAIN_API_KEY = os.getenv("BRAIN_API_KEY", "")

ALLOWED_PATHS = {
    "/health",
    "/register",
    "/heartbeat",
    "/action",
    "/experience",
    "/evaluation",
    "/save",
    "/stats",
    "/diagnose",
}

app = FastAPI(title="V53 Privacy Relay")


# ---------------------------------------------------------
# PRIVACY SANITIZER
# ---------------------------------------------------------

BLOCKED_KEYS = {
    "ip",
    "ip_address",
    "public_ip",
    "local_ip",
    "remote_ip",
    "hostname",
    "host",
    "machine",
    "machine_name",
    "computer_name",
    "username",
    "user",
    "home",
    "home_dir",
    "cwd",
    "path",
    "file_path",
    "gps",
    "latitude",
    "longitude",
    "location",
    "device_id",
    "fingerprint",
    "mac",
    "mac_address",
    "cookies",
    "user_agent",
}


def sanitize(value: Any):
    if isinstance(value, dict):
        return {
            k: sanitize(v)
            for k, v in value.items()
            if k.lower() not in BLOCKED_KEYS
        }

    if isinstance(value, list):
        return [sanitize(v) for v in value]

    return value


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "V53 Privacy Relay",
    }


# ---------------------------------------------------------
# AUTH
# ---------------------------------------------------------

def check_relay_auth(authorization: str | None):
    if not RELAY_API_KEY:
        return

    expected = f"Bearer {RELAY_API_KEY}"

    if authorization != expected:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )


# ---------------------------------------------------------
# FORWARDER
# ---------------------------------------------------------

@app.api_route(
    "/forward/{path:path}",
    methods=["GET", "POST"],
)
async def forward(
    path: str,
    request: Request,
    authorization: str | None = Header(default=None),
):
    check_relay_auth(authorization)

    if not UPSTREAM:
        raise HTTPException(
            status_code=500,
            detail="BRAIN_UPSTREAM_URL is not configured",
        )

    upstream_path = "/" + path.lstrip("/")

    if upstream_path not in ALLOWED_PATHS:
        raise HTTPException(
            status_code=403,
            detail="Path not allowed",
        )

    # -----------------------------------------------------
    # GET REQUEST
    # -----------------------------------------------------

    if request.method == "GET":
        body = None

    # -----------------------------------------------------
    # POST REQUEST
    # -----------------------------------------------------

    else:
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Expected JSON body",
            )

        # SECOND privacy sanitization
        payload = sanitize(payload)

        body = payload

    # -----------------------------------------------------
    # FRESH UPSTREAM REQUEST
    # -----------------------------------------------------

    headers = {
        "Content-Type": "application/json",
    }

    if BRAIN_API_KEY:
        headers["Authorization"] = f"Bearer {BRAIN_API_KEY}"

    upstream_url = f"{UPSTREAM}{upstream_path}"

    async with httpx.AsyncClient(timeout=20.0) as client:

        if request.method == "GET":
            response = await client.get(
                upstream_url,
                headers=headers,
            )

        else:
            response = await client.post(
                upstream_url,
                json=body,
                headers=headers,
            )

    # Don't blindly forward arbitrary upstream headers.
    content_type = response.headers.get(
        "content-type",
        "application/json",
    )

    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=content_type.split(";")[0],
    )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000"))
    )