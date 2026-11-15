from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder 
from feed_agents import run

sub_app = FastAPI(title="Sub App", version="1.0.0")

@sub_app.get("/health_sub")
async def sub_health():
    return {"ok": True}

@sub_app.get("/feedagents")
async def feed_agents():
    result = await run_in_threadpool(run, "yes")
    return {"ok": True, "result": jsonable_encoder(result)}