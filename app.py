import os, tempfile
import streamlit as st
from ingest import Store
from agent import build_graph

st.set_page_config(page_title="DataPulse AI", page_icon="📊", layout="wide")
USER, PWD = os.getenv("DEMO_USER", "demo"), os.getenv("DEMO_PASS", "DataPulse2026")

if not st.session_state.get("auth"):
    st.title("📊 DataPulse AI")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Log in"):
        if u == USER and p == PWD:
            st.session_state.auth = True
            st.rerun()
        st.error("Wrong credentials")
    st.stop()


@st.cache_resource
def graph():
    return build_graph()


if "store" not in st.session_state:
    st.session_state.store, st.session_state.msgs, st.session_state.seen = Store(), [], set()
store = st.session_state.store

with st.sidebar:
    st.header("Data")
    if st.button("Load sample sales.csv"):
        store.add_file("sample_data/sales.csv")
    for f in st.file_uploader("CSV, Parquet or PDF", type=["csv", "parquet", "pdf"],
                              accept_multiple_files=True) or []:
        if (f.name, f.size) in st.session_state.seen:
            continue
        with st.spinner(f"Ingesting {f.name}..."):
            path = os.path.join(tempfile.mkdtemp(), f.name)
            with open(path, "wb") as fh:
                fh.write(f.getbuffer())
            store.add_file(path)
        st.session_state.seen.add((f.name, f.size))
    for n, df in store.tables.items():
        st.caption(f"🗂 {n} ({len(df)} rows)")
    if store.chunks:
        st.caption(f"📄 {len(store.chunks)} PDF chunks indexed")


def render(m, key):
    st.markdown(m["content"])
    if m.get("trace"):
        with st.expander(f"SQL ({len(m['trace'])} attempt(s))"):
            for t in m["trace"]:
                st.code(t["sql"], language="sql")
                if t["error"]:
                    st.caption(f"❌ {t['error']}")
    if m.get("df") is not None:
        st.dataframe(m["df"], use_container_width=True)
    if m.get("fig") is not None:
        st.plotly_chart(m["fig"], use_container_width=True, key=key)
    if m.get("chunks"):
        with st.expander("Sources"):
            for i, c in enumerate(m["chunks"], 1):
                st.markdown(f"**[{i}] {c['source']}**\n\n{c['text'][:500]}")


st.title("📊 DataPulse AI")
for i, m in enumerate(st.session_state.msgs):
    with st.chat_message(m["role"]):
        render(m, f"fig{i}")

if q := st.chat_input("Ask about your data and reports..."):
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                out = graph().invoke({"question": q, "store": store, "retries": 0, "trace": []})
                msg = {"role": "assistant", "content": out["answer"], "trace": out.get("trace"),
                       "df": out.get("df"), "fig": out.get("fig"), "chunks": out.get("chunks")}
            except Exception as e:
                msg = {"role": "assistant", "content": f"Error: {e}"}
        render(msg, f"fig{len(st.session_state.msgs) + 1}")
    st.session_state.msgs += [{"role": "user", "content": q}, msg]
