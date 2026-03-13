import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    if os.path.getsize(path) == 0:
        raise ValueError("PDF file is empty")

    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    return chunks