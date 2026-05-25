import streamlit as st
import os
import sys

#Add backend path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..","backend")

    )
)

from pdf_loader import load_documents
from text_splitter import split_documents
from vector_store import create_vector_store, save_vector_store
from llm_chain import generate_answer

#page config
st.set_page_config(
    page_title = "Enterprise Knowledge Assistant",
    page_icon= "📄",
    layout= "wide"
)
#sidebar

with st.sidebar:
    st.title("📄 Enterprise RAG Agent")

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload your TXT/PDF file",
        type = ["pdf","txt"]
    )
    st.markdown("---")

    st.info(
        "Upload company documents, reports, policies, "
        "contracts, or research papers and ask questions instantly."
    )

#Main Page
st.title("AI-Powered Document Intelligence System")

st.markdown(

    "Ask questions from your uploaded documents using "
    "Retrieval-Augmented Generation (RAG)."
)
st.markdown("---")

#Process PDF
if uploaded_file is not None:

    #Save uploaded file temporarily
    file_path = os.path.join(
        "data/sample_pdfs",
        uploaded_file.name
    )
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Processing your file..."):
        try:
            #load PDF
            docs = load_documents(file_path)

            #chunking
            chunks = split_documents(docs)

            #create vector store
            vectorstore = create_vector_store(chunks)

            #save locally
            save_vector_store(vectorstore)

        except Exception as e:
            st.error(f"Error: {str(e)}")
    #Ask question
    query = st.text_input(
        "Ask your question from the document:"
    )
    if query:
        with st.spinner("Generating answer..."):
            try:
                answer = generate_answer(query)

                st.markdown("## Final Answer")
                st.write(answer)

            except Exception as e:
                st.error(f"Error: {str(e)}")

