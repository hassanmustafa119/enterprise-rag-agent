import streamlit as st
import os
import sys

# Add backend path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "backend")
    )
)

from pdf_loader import load_documents
from text_splitter import split_documents
from vector_store import create_vector_store, save_vector_store
from llm_chain import generate_answer


#Page config
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📄",
    layout="wide"
)

#Sidebar
with st.sidebar:
    st.title("📄 Enterprise RAG Agent")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload your TXT/PDF file",
        type=["pdf", "txt"]
    )

    st.markdown("---")

    st.info(
        "Upload company documents, reports, policies, "
        "contracts, or research papers and ask questions instantly."
    )


#Main page
st.title("AI-Powered Document Intelligence System")

st.markdown(
    "Ask questions from your uploaded documents using "
    "Retrieval-Augmented Generation (RAG)."
)

st.markdown("---")


#Process uploaded file
if uploaded_file is not None:

    #Safe directory creation
    upload_dir = "data/sample_pdfs"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, uploaded_file.name)

    #Save file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing your file..."):
        try:
            # Load document
            docs = load_documents(file_path)

            # Chunking
            chunks = split_documents(docs)

            # Create vector store
            vectorstore = create_vector_store(chunks)

            
            save_vector_store(vectorstore)

        except Exception as e:
            st.error(f"Error: {str(e)}")



    query = st.text_input("Ask your question from the document:")

    if query:
        with st.spinner("Generating answer..."):
            try:
                answer = generate_answer(query)

                st.markdown("## Final Answer")
                st.write(answer)

            except Exception as e:
                st.error(f"Error: {str(e)}")