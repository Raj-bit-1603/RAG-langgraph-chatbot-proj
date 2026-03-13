from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_vector_db():

    loader = TextLoader("data.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    documents = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vector_db = FAISS.from_documents(
        documents,
        embeddings
    )

    print("Documents in Vector DB:", len(documents))

    return vector_db