from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
import os

load_dotenv()


def get_embeddings():
    """
    Initialize Gemini Embeddings
    """

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
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

    vectorstore.save_local(path)


def load_vector_store(
    path="vectorstore/faiss_index"
):
    """
    Load FAISS vector store from disk
    """

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

        # Load document
        docs = load_documents(file_path)

        # Split into chunks
        chunks = split_documents(docs)

        # Create vector DB
        vectorstore = create_vector_store(chunks)

        # Save vector DB
        save_vector_store(vectorstore)

        print("FAISS Vector Store created successfully.")

        print(f"Total chunks stored: {len(chunks)}")

    except Exception as e:

        print(f"Error: {str(e)}")