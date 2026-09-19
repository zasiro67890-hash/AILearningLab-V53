import os

from fastapi import FastAPI, HTTPException, Request
from agent import BrainHub

app = FastAPI(title="V53 Secure Autonomous MindMesh")

memory = {}
memory_b = {}
route_memory = {}

brain = BrainHub(memory, route_memory, memory_b)


@app.get("/")
async def root():
    return {
        "service": "V53 Secure Autonomous MindMesh",
        "status": "online"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "V53 BrainHub"
    }


@app.get("/stats")
async def stats():
    return brain.snapshot()


def authorized(request: Request):
    expected = os.getenv("BRAIN_API_KEY")

    if not expected:
        return True

    auth = request.headers.get("authorization", "")

    return auth == f"Bearer {expected}"


@app.post("/register")
async def register(request: Request):
    if not authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    payload = await request.json()

    return brain.register(
        payload.get("worker_id")
    )


@app.post("/heartbeat")
async def heartbeat(request: Request):
    if not authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    payload = await request.json()

    return brain.heartbeat(
        payload.get("worker_id"),
        payload.get("info")
    )


@app.post("/action")
async def action(request: Request):
    if not authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    payload = await request.json()

    return brain.choose(
        payload.get("worker_id"),
        payload.get("state", {}),
        payload.get("events", [])
    )


@app.post("/experience")
async def experience(request: Request):
    if not authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    payload = await request.json()

    return brain.experience(
        payload.get("worker_id"),
        payload
    )


@app.post("/evaluation")
async def evaluation(request: Request):
    if not authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    payload = await request.json()

    return brain.record_evaluation(
        payload.get("worker_id"),
        payload.get("reward", 0.0),
        payload.get("level", 1),
        payload.get("goals", 0)
    )


@app.post("/save")
async def save(request: Request):
    if not authorized(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    brain.save()

    return {"ok": True}