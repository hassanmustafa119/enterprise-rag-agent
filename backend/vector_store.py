from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()


#EMBEDDINGS
def get_embeddings():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing in environment variables")

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )


#Create
def create_vector_store(chunks):
    embeddings = get_embeddings()

    return FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )


#Save
def save_vector_store(vectorstore, path="vectorstore/faiss_index"):
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)


#Load and safe
def load_vector_store(path="vectorstore/faiss_index"):
    """
    Safe FAISS loader (prevents Streamlit crash if missing index)
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FAISS index not found at: {path}. "
            "Please run vector store creation script first."
        )

    embeddings = get_embeddings()

    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )


if __name__ == "__main__":

    from pdf_loader import load_documents
    from text_splitter import split_documents

    try:
        docs = load_documents("data/sample.pdf")
        chunks = split_documents(docs)

        vs = create_vector_store(chunks)
        save_vector_store(vs)

        print("FAISS Vector Store created successfully")
        print(f"Chunks: {len(chunks)}")

    except Exception as e:
        print(f"Error: {e}")