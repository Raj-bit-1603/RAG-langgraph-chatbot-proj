
import streamlit as st

from backend.ingestion import load_documents
from backend.embeddings import create_vector_store
from workflow.graph import build_graph

# from langchain_community.chat_models import ChatOllama
#
# llm = ChatOllama(
#     model="llama3"
# )
# from langchain_community.chat_models import ChatOllama
#
# llm = ChatOllama(
#     model="phi3:mini"
# )
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(
    model="gemma:2b"
)

import os

file_path = "data/docs/ai_intro.pdf"

if not os.path.exists(file_path):
    st.error("PDF file not found. Please add a document in data/docs folder.")
    st.stop()

# docs = load_documents(file_path)

st.title("RAG Chatbot with LangGraph")

docs = load_documents("data/docs/ai_intro.pdf")

vector_db = create_vector_store(docs)
#
# llm = ChatOpenAI(
#     temperature=0,
#     model="gpt-3.5-turbo"
# )

graph = build_graph()
#
# query = st.chat_input("Ask something...")
#
# if query:
#
#     state = {
#         "query": query,
#         "vector_db": vector_db,
#         "llm": llm
#     }
#
#     result = graph.invoke(state)
#
#     st.write(result["final_response"])

query = st.chat_input("Ask something...")

if query:

    state = {
        "query": query,
        "vector_db": vector_db,
        "llm": llm
    }

    result = graph.invoke(state)

    st.write(result["final_response"])
