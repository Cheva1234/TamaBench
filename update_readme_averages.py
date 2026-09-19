import sqlite3
import re

DB_PATH = "tamabench_results.db"

def calculate_averages():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    query = """
    SELECT 
        runs.model_name,
        COUNT(*) as episodes,
        ROUND(AVG(outcomes.survived)*100, 1) as survival_rate,
        ROUND(AVG(outcomes.simulated_days), 2) as avg_days,
        ROUND(AVG(outcomes.final_health), 1) as avg_health,
        ROUND(AVG(outcomes.total_income)) as avg_income,
        ROUND(AVG(outcomes.total_spending)) as avg_spending
    FROM runs
    JOIN outcomes ON runs.run_id = outcomes.run_id
    
    GROUP BY runs.model_name
    """
    
    c.execute(query)
    rows = c.fetchall()
    
    query_tokens = """
    SELECT 
        runs.model_name,
        ROUND(AVG(output_tokens)) as avg_out_tokens
    FROM runs
    JOIN (
        SELECT run_id, SUM(output_tokens) as output_tokens FROM runtime_metrics GROUP BY run_id
    ) rm ON runs.run_id = rm.run_id
    
    GROUP BY runs.model_name
    """
    
    c.execute(query_tokens)
    tokens_map = {r[0]: r[1] for r in c.fetchall()}
    conn.close()
    
    results = []
    for r in rows:
        model = r[0]
        episodes = r[1]
        surv = f"{r[2]}%"
        days = r[3]
        health = r[4]
        income = r[5]
        spending = r[6]
        tokens = tokens_map.get(model, 0)
        
        score = int((days * 1000) + (health * 10) + (income - spending))
        
        results.append({
            "model": model,
            "episodes": episodes,
            "survival": surv,
            "score": score,
            "days": days,
            "tokens": tokens
        })
        
    return results

def update_readme(results):
    results.sort(key=lambda x: x["score"], reverse=True)
    
    table = "## Current Results (5-Episode Averages)\n\n"
    table += "| Model | Episodes | Survival | Avg Score | Avg Days | Avg Output Tokens |\n"
    table += "|---|---:|---:|---:|---:|---:|\n"
    
    for r in results:
        table += f"| `{r['model']}` + Harness V1 | {r['episodes']} | {r['survival']} | {r['score']:,} | {r['days']} | {r['tokens']:,} |\n"
        
    table += "\n> **Note**: Averages calculated across 5 episodes to account for RNG variance in sickness and dynamic economy constraints.\n"

    with open("README.md", "r") as f:
        content = f.read()
        
    content = re.sub(r"## Current Results \(5-Episode Averages\).*?\n\n---\n", table + "\n---\n", content, flags=re.DOTALL)
    
    with open("README.md", "w") as f:
        f.write(content)

if __name__ == "__main__":
    res = calculate_averages()
    update_readme(res)
    print("README updated with averages.")
