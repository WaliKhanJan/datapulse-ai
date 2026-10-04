import os, re, json
from typing import Any, TypedDict
import plotly.express as px
from openai import OpenAI
from langgraph.graph import StateGraph, END

MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-32b")
MAX_RETRIES = 3
_client = None


def llm(system, user):
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=os.environ["LLM_API_KEY"],
        )
    r = _client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()


# Public demo: allow only a single read-only query over the loaded tables.
BLOCKED = re.compile(
    r"\b(read_\w+|glob|copy|attach|install|load|pragma|export|create|drop|insert|update|delete|alter)\b"
    r"|\b(from|join)\s*\(?\s*['\"]", re.I)


def safe(sql):
    s = sql.strip().rstrip(";")
    if ";" in s or not re.match(r"(?i)^(select|with)\b", s) or BLOCKED.search(s):
        raise ValueError("Only a single read-only SELECT over the loaded tables is allowed.")
    return s


class S(TypedDict, total=False):
    question: str
    store: Any
    route: str
    sql: str
    error: str
    retries: int
    trace: list
    df: Any
    chunks: list
    answer: str
    fig: Any


def classify(s):
    st = s["store"]
    has_t, has_d = bool(st.tables), bool(st.chunks)
    if not (has_t or has_d):
        return {"route": "none"}
    if has_t != has_d:
        return {"route": "sql" if has_t else "docs"}
    r = llm("Classify the question. Reply with one word: sql (needs numbers from tables), "
            "docs (needs text from reports), or both.",
            f"Tables:\n{st.schema()}\n\nQuestion: {s['question']}")
    r = (r.split() or [""])[0].strip(".,").lower()
    return {"route": r if r in ("sql", "docs", "both") else "both"}


def write_sql(s):
    prompt = f"Schema:\n{s['store'].schema()}\n\nQuestion: {s['question']}"
    if s.get("error"):
        prompt += f"\n\nYour previous SQL failed.\nSQL: {s['sql']}\nError: {s['error']}\nFix it."
    out = llm("You write DuckDB SQL. Return ONLY one SELECT query, no markdown, no explanation. "
              "Use only tables and columns from the schema.", prompt)
    return {"sql": re.sub(r"```(?:sql)?", "", out).strip()}


def run_sql(s):
    trace = s.get("trace", [])
    try:
        df = s["store"].con.execute(safe(s["sql"])).df().head(200)
        return {"df": df, "error": "", "trace": trace + [{"sql": s["sql"], "error": ""}]}
    except Exception as e:
        return {"error": str(e), "retries": s.get("retries", 0) + 1,
                "trace": trace + [{"sql": s["sql"], "error": str(e)}]}


def retrieve(s):
    return {"chunks": s["store"].search(s["question"])}


def make_chart(df, q):
    try:
        raw = llm('Pick a chart. Reply JSON only: {"type":"bar|line|none","x":"col","y":"col"}. '
                  "Columns must exist.",
                  f"Question: {q}\nColumns: {list(df.columns)}\n{df.head(3).to_string()}")
        spec = json.loads(re.search(r"\{.*\}", raw, re.S).group())
        if spec["type"] == "none" or spec["x"] not in df or spec["y"] not in df:
            return None
        return (px.line if spec["type"] == "line" else px.bar)(df, x=spec["x"], y=spec["y"])
    except Exception:
        return None


def synthesize(s):
    if s["route"] == "none":
        return {"answer": "Upload a CSV/Parquet file or a PDF first (or load the sample data)."}
    df, parts = s.get("df"), []
    if df is not None and not s.get("error"):
        parts.append(f"SQL RESULT (query: {s['sql']}):\n{df.head(30).to_string(index=False)}")
    elif s.get("error"):
        parts.append(f"The SQL query failed after retries: {s['error']}")
    for i, c in enumerate(s.get("chunks") or [], 1):
        parts.append(f"[{i}] ({c['source']}) {c['text']}")
    ans = llm("You are a data analyst. Answer ONLY from the provided SQL result and excerpts. "
              "Cite document excerpts as [n]. If the data is missing or the query failed, say so "
              "instead of guessing. Be concise.",
              f"Question: {s['question']}\n\n" + "\n\n".join(parts))
    ok = df is not None and not s.get("error") and len(df) > 1
    return {"answer": ans, "fig": make_chart(df, s["question"]) if ok else None}


def after_classify(s):
    return {"sql": "write_sql", "both": "write_sql", "docs": "retrieve"}.get(s["route"], "synthesize")


def after_sql(s):
    if s.get("error") and s["retries"] <= MAX_RETRIES:
        return "write_sql"
    return "retrieve" if s["route"] == "both" else "synthesize"


def build_graph():
    g = StateGraph(S)
    for name, fn in [("classify", classify), ("write_sql", write_sql), ("run_sql", run_sql),
                     ("retrieve", retrieve), ("synthesize", synthesize)]:
        g.add_node(name, fn)
    g.set_entry_point("classify")
    g.add_conditional_edges("classify", after_classify,
                            {"write_sql": "write_sql", "retrieve": "retrieve", "synthesize": "synthesize"})
    g.add_edge("write_sql", "run_sql")
    g.add_conditional_edges("run_sql", after_sql,
                            {"write_sql": "write_sql", "retrieve": "retrieve", "synthesize": "synthesize"})
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()
