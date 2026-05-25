from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
import os

load_dotenv()


def get_embeddings():
    """
    Initialize Gemini Embeddings
    """
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing in environment variables")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    return embeddings


def create_vector_store(chunks):
    """
    Convert chunks into embeddings and store in FAISS
    """
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


def save_vector_store(
    vectorstore,
    path="vectorstore/faiss_index"
):
    """
    Save FAISS vector store locally
    """
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)


def load_vector_store(
    path="vectorstore/faiss_index"
):
    """
    Load FAISS vector store from disk (SAFE)
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            "FAISS index not found. You must create and save it first."
        )

    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


if __name__ == "__main__":

    from pdf_loader import load_documents
    from text_splitter import split_documents

    file_path = "data/sample.pdf"

    try:
        docs = load_documents(file_path)
        chunks = split_documents(docs)

        vectorstore = create_vector_store(chunks)
        save_vector_store(vectorstore)

        print("FAISS Vector Store created successfully.")
        print(f"Total chunks stored: {len(chunks)}")

    except Exception as e:
        print(f"Error: {str(e)}")