import asyncio
import sys
from pathlib import Path
from pathlib import Path

# make `core` and `utils` importable when Streamlit runs this file from the ui/ folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from core.pipeline import RecruitmentPipeline  # noqa: E402
from utils.schemas import JobDescription  # noqa: E402

st.set_page_config(page_title="HireFlow", page_icon="🧑‍💼", layout="wide")


@st.cache_resource(show_spinner="Starting HireFlow...")


def ensure_event_loop():
    """Streamlit runs the script in a worker thread that has no asyncio event loop,
    but the Google client libraries need one when they are created."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


ensure_event_loop()
def get_pipeline() -> RecruitmentPipeline:
    """Created once and shared across reruns, so the indexes are not reloaded on every click."""
    return RecruitmentPipeline()


class HireFlowUI:
    def __init__(self, pipeline: RecruitmentPipeline):
        self.p = pipeline
        st.session_state.setdefault("page", "main")
        st.session_state.setdefault("results", [])
        st.session_state.setdefault("evals", {})

    # ---------------------------------------------------------------- sidebar
    def sidebar(self):
        st.sidebar.title("🧑‍💼 HireFlow")
        status = self.p.status()
        if status["hybrid_ready"]:
            st.sidebar.success("System ready (BM25 + FAISS)")
        elif status["bm25_ready"]:
            st.sidebar.warning("Keyword search only - vector search unavailable (check key / quota)")
        else:
            st.sidebar.error("No resumes indexed yet")
        llm_text = "available" if status["llm_available"] else "unavailable (rule-based fallback)"
        st.sidebar.caption(f"Gemini: {llm_text}")
        st.sidebar.metric("Resumes indexed", status["bm25_resumes"])
        st.sidebar.metric("FAISS vectors", status["vector_stats"]["total_vector_count"])
        st.sidebar.divider()
        mem = status["memory"]
        st.sidebar.write(f"**Memory** - searches: {mem['search_count']}, views: {mem['candidate_views']}")
        if st.sidebar.button("🔎 Main page", use_container_width=True):
            st.session_state.page = "main"
        if st.sidebar.button("📊 Memory & evaluation", use_container_width=True):
            st.session_state.page = "eval"

    # ----------------------------------------------------------------- upload
    def upload_section(self):
        with st.expander("📤 Upload resumes (PDF)"):
            files = st.file_uploader("Choose PDF resumes", type="pdf", accept_multiple_files=True)
            if files and st.button("Process and index"):
                for f in files:
                    with st.spinner(f"Processing {f.name}..."):
                        meta = self.p.add_uploaded_resume(f.getvalue(), f.name)
                    if meta:
                        st.success(f"Indexed {meta.get('name', f.name)} ({meta.get('title', '')})")
                    else:
                        st.warning(f"No text could be extracted from {f.name}")

    # ----------------------------------------------------------------- search
    def search_section(self):
        st.subheader("Find candidates")
        with st.form("search_form"):
            c1, c2 = st.columns([2, 1])
            title = c1.text_input("Job title", "Senior Accountant")
            mode = c2.selectbox(
                "Search mode",
                ["auto", "shallow", "deep"],
                help="shallow = vectors only (fast); deep = multi-query + hybrid + AI re-rank",
            )
            description = st.text_area(
                "Job description / requirements",
                "Looking for an accountant with financial reporting, month-end close "
                "and QuickBooks experience.",
                height=100,
            )
            c3, c4, c5, c6 = st.columns(4)
            skills_raw = c3.text_input("Required skills (comma separated)", "")
            location = c4.text_input("Location contains", "")
            min_exp = c5.number_input("Min. years of experience", 0, 40, 0)
            top_k = c6.slider("Results", 1, 15, 5)
            ai_eval = st.checkbox("Add AI evaluation (fit score)", value=True)
            submitted = st.form_submit_button("Search", type="primary")

        if submitted:
            skills = [s.strip() for s in skills_raw.split(",") if s.strip()]
            query = f"{title}. {description}".strip()
            with st.spinner("Searching..."):
                results = self.p.search(
                    query,
                    top_k=top_k,
                    search_mode=mode,
                    required_skills=skills,
                    locations=[location] if location else None,
                    min_experience=min_exp or None,
                    jd_context=description,
                )
            st.session_state.results = results
            st.session_state.evals = {}
            if ai_eval and results:
                jd = JobDescription(
                    jd_id="ui",
                    title=title,
                    text=description,
                    required_skills=skills,
                    min_experience=min_exp or None,
                    location=location or None,
                )
                with st.spinner("AI is evaluating the candidates..."):
                    evals = self.p.evaluate_candidates(results, jd)
                st.session_state.evals = {e.candidate_id: e for e in evals}

        self.show_results()

    def show_results(self):
        results = st.session_state.results
        if not results:
            st.info("Run a search to see candidates.")
            return

        evals = st.session_state.evals
        st.caption(f"Mode used: **{results[0]['search_mode']}** | queries: {results[0]['queries_used']}")

        if evals:  # best AI fit score first
            def fit(r):
                ev = evals.get(r["candidate_id"])
                return ev.fit_score if ev else -1

            results = sorted(results, key=fit, reverse=True)

        for i, r in enumerate(results, start=1):
            m = r["metadata"]
            ev = evals.get(r["candidate_id"])
            header = f"{i}. {r['name']} - {m.get('title', '')}"
            if ev:
                header += f"  |  Fit {ev.fit_score}/100"
            with st.expander(header, expanded=(i == 1)):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Match score", r["combined_score"])
                c2.metric("Keyword (BM25)", r["bm25_score"])
                c3.metric("Meaning (vector)", r["vector_score"])
                c4.metric("Experience", f"{m.get('experience', '?')} yrs")
                st.write(f"📍 {m.get('location', 'n/a')}   ✉️ {m.get('email', 'n/a')}   📞 {m.get('phone', 'n/a')}")
                st.write("**Skills:** " + ", ".join(m.get("skills", [])))

                if ev:
                    method = ev.evidence.get("method", "") if ev.evidence else ""
                    st.progress(ev.fit_score / 100, text=f"AI fit score: {ev.fit_score}/100 ({method})")
                    st.write(f"**Summary:** {ev.summary}")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.write("**Strengths**")
                    for s in ev.strengths:
                        col_a.write(f"- {s}")
                    col_b.write("**Gaps**")
                    for s in ev.gaps:
                        col_b.write(f"- {s}")
                    col_c.write("**Risks**")
                    for s in ev.risks or ["none flagged"]:
                        col_c.write(f"- {s}")

                if st.button("👁 View full resume", key=f"view_{r['candidate_id']}"):
                    self.p.memory.record_candidate_view(r["name"])
                    st.text(r["text"])

    # ------------------------------------------------------ memory & evaluation
    def eval_page(self):
        st.header("Memory & evaluation")
        memory = self.p.memory
        stats = memory.get_memory_stats()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total messages", stats["total_messages"])
        c2.metric("Searches", stats["search_count"])
        c3.metric("Candidate views", stats["candidate_views"])

        left, right = st.columns(2)
        left.write("**Recent searches**")
        for q in reversed(memory.get_search_history()):
            left.write(f"- {q[:120]}")
        right.write("**Recently viewed candidates**")
        for name in reversed(memory.get_recent_views()):
            right.write(f"- {name}")
        if st.button("Clear memory"):
            memory.clear()
            st.rerun()

        st.divider()
        st.subheader("Search quality evaluation")
        st.caption(
            "RAGAS uses Gemini as a judge and makes several API calls. "
            "Without it, simple heuristic metrics are used (labelled 'heuristic')."
        )
        with st.form("eval_form"):
            q = st.text_input("Test query", "Senior accountant with financial reporting experience")
            skills = st.text_input("Expected skills (comma separated)", "Financial Reporting, Variance Analysis")
            mode = st.selectbox("Search mode", ["deep", "shallow"])
            use_ragas = st.checkbox("Use RAGAS (slower, uses the LLM)", value=False)
            go = st.form_submit_button("Run evaluation", type="primary")

        if go:
            expected = [s.strip() for s in skills.split(",") if s.strip()]
            with st.spinner("Evaluating..."):
                m = self.p.evaluator.evaluate_search_quality(
                    q, expected, search_mode=mode, use_ragas=use_ragas
                )
            st.caption(f"Method: **{m.method}**")
            cols = st.columns(5)
            labels = [
                ("Answer relevancy", m.answer_relevancy),
                ("Context precision", m.context_precision),
                ("Faithfulness", m.faithfulness),
                ("Answer correctness", m.answer_correctness),
                ("Overall", m.overall_score),
            ]
            for col, (label, value) in zip(cols, labels):
                col.metric(label, f"{value:.2f}")

        history = self.p.evaluator.evaluation_history
        if history:
            st.write("**Evaluation history**")
            st.dataframe(pd.DataFrame(history), use_container_width=True)
            if st.button("Export to CSV"):
                st.success(f"Saved to {self.p.evaluator.export_evaluations()}")

    # -------------------------------------------------------------------- run
    def run(self):
        self.sidebar()
        if st.session_state.page == "eval":
            self.eval_page()
        else:
            st.title("HireFlow: AI-powered recruitment")
            self.upload_section()
            self.search_section()


pipeline = get_pipeline()
if not pipeline.is_indexed():  # first run: index the bundled resumes automatically
    with st.spinner("First run: indexing resumes (embeddings are cached for next time)..."):
        pipeline.ingest_resumes()

HireFlowUI(pipeline).run()