from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import os
from contextlib import asynccontextmanager

DB_PATH = "leaderboard.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE,
            agent_name TEXT,
            survived BOOLEAN,
            simulated_days REAL,
            avg_health REAL,
            score REAL
        )
    ''')
    conn.commit()
    conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="TamaBench Leaderboard API", lifespan=lifespan)

class ScoreSubmit(BaseModel):
    run_id: str
    agent_name: str
    survived: bool
    simulated_days: float
    avg_health: float
    score: float

@app.post("/submit")
def submit_score(score: ScoreSubmit):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO leaderboard (run_id, agent_name, survived, simulated_days, avg_health, score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (score.run_id, score.agent_name, score.survived, score.simulated_days, score.avg_health, score.score))
        conn.commit()
        conn.close()
        return {"status": "success", "message": "Score submitted"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Run ID already exists")

@app.get("/leaderboard")
def get_leaderboard(limit: int = 10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT agent_name, survived, simulated_days, avg_health, score
        FROM leaderboard 
        ORDER BY score DESC 
        LIMIT ?
    ''', (limit,))
    rows = c.fetchall()
    conn.close()
    
    result = []
    for r in rows:
        result.append({
            "agent_name": r[0],
            "survived": bool(r[1]),
            "simulated_days": r[2],
            "avg_health": r[3],
            "score": r[4]
        })
    return {"leaderboard": result}
