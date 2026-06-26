"""Streamlit chat UI for PharmaLens."""

import os

import requests
import streamlit as st

API_URL = os.getenv("PHARMALENS_API_URL", "http://localhost:8000")
st.set_page_config(page_title="PharmaLens", page_icon="💊", layout="wide")
st.title("💊 PharmaLens")
st.caption("Pharmaceutical Product Lifecycle Knowledge System")

with st.sidebar:
    st.header("Knowledge Base")
    try:
        st.metric("Indexed Chunks", requests.get(f"{API_URL}/collection/stats", timeout=3).json()["total_chunks"])
    except Exception:
        st.warning("API not reachable")
    st.divider()
    reg_body = st.selectbox("Regulatory Body", ["Any", "FDA", "EMA", "ICH"])
    formulation = st.selectbox("Formulation", ["Any", "SubcutaneousSolution", "OralTablet"])
    doc_type = st.selectbox("Document Type", ["Any", "PSG", "EPAR", "CitizenPetition"])
    st.divider()
    for example in ("What bioequivalence studies are required for generic oral semaglutide?",
                    "What are the contraindications for Ozempic?",
                    "What is the approved posology for Wegovy in adults?"):
        if st.button(example, use_container_width=True):
            st.session_state.pending_query = example

st.session_state.setdefault("messages", [])
st.session_state.setdefault("history", [])


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 {len(sources)} Sources"):
        for source in sources:
            st.markdown(f"**{source['filename']}** | §{source['section_number']} {source['section']} | p.{source['page']} | {source['regulatory_body']} {source['doc_type']} | Score: {source['score']}")
            st.caption(source["excerpt"])
            st.divider()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))
        for alert in message.get("delta_alerts", []):
            st.info(alert)

prompt = st.chat_input("Ask about any pharma product...") or st.session_state.pop("pending_query", None)
if prompt:
    filters = {key: value for key, value in {
        "regulatory_body": reg_body, "formulation": formulation, "doc_type": doc_type}.items() if value != "Any"}
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            response = requests.post(f"{API_URL}/query", json={"query": prompt,
                "conversation_history": st.session_state.history, "metadata_filters": filters}, timeout=90)
            response.raise_for_status()
            data = response.json()
            st.markdown(data["answer"])
            show_sources(data.get("sources", []))
            for alert in data.get("delta_alerts", []):
                st.info(alert)
            st.session_state.messages.append({"role": "assistant", "content": data["answer"],
                "sources": data.get("sources", []), "delta_alerts": data.get("delta_alerts", [])})
            st.session_state.history.extend(({"role": "user", "content": prompt}, {"role": "assistant", "content": data["answer"]}))
        except Exception as exc:
            st.error(f"Error: {exc}")
