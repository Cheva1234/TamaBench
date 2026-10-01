"""Local unverified-submission prototype, not a trusted public leaderboard."""
from contextlib import asynccontextmanager, closing
import os
import sqlite3

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

DB_PATH = os.environ.get("TAMABENCH_LEADERBOARD_DB", "leaderboard.db")


def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT UNIQUE,
            agent_name TEXT, survived BOOLEAN, simulated_days REAL,
            avg_health REAL, score REAL)''')


@asynccontextmanager
async def lifespan(app):
    init_db()
    yield


app = FastAPI(title="TamaBench local unverified submissions", lifespan=lifespan,
    description="Client-supplied results are unverified. This prototype has no authentication or trajectory verification and must not be presented as a trusted public ranking.")


class ScoreSubmit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(min_length=1, max_length=128)
    agent_name: str = Field(min_length=1, max_length=256)
    survived: bool
    simulated_days: FiniteFloat = Field(ge=0)
    avg_health: FiniteFloat = Field(ge=0, le=100)
    score: FiniteFloat


@app.post("/submit")
def submit_score(score: ScoreSubmit):
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn, conn:
            conn.execute('''INSERT INTO leaderboard
                (run_id,agent_name,survived,simulated_days,avg_health,score) VALUES (?,?,?,?,?,?)''',
                (score.run_id,score.agent_name,score.survived,score.simulated_days,score.avg_health,score.score))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Run ID already exists")
    return {"status":"success", "verification_status":"unverified", "message":"Stored client-supplied result; no benchmark verification performed"}


@app.get("/leaderboard")
def get_leaderboard(limit: int = Query(10, ge=1, le=100)):
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute('''SELECT agent_name,survived,simulated_days,avg_health,score
                               FROM leaderboard ORDER BY score DESC LIMIT ?''', (limit,)).fetchall()
    return {"verification_status":"unverified", "leaderboard":[dict(zip(
        ("agent_name","survived","simulated_days","avg_health","score"),
        (r[0],bool(r[1]),r[2],r[3],r[4]))) for r in rows]}
