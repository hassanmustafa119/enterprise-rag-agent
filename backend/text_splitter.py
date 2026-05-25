from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def split_documents(documents):
    """
    Splits PDF documents into optimized chunks for RAG

    Args:
        documents (list): List of Langchain document objects
    
    Returns:
         chunks (list): List of chunked document objects
    """

    # Initialize Smart Text Splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        length_function = len,
        separators = ["\n\n","\n","."," ",""]
    )

    #Split documents into chunks
    chunks = splitter.split_documents(documents)

    #Validation
    if not chunks:
        raise ValueError("Chunking failed: No chunks were created.")
    
    return chunks

if __name__ == "__main__":
    from backend.pdf_loader import load_pdf

    pdf_path = "data/sample_pdfs/sample.pdf"

    try:
        docs = load_pdf(pdf_path)
        chunks = split_documents(docs)

        print(f"Total chunks created :{len(chunks)}\n")

        print("sample chunk:\n")
        print(chunks[0].page_content[:1000])
    except Exception as e:
        print(f"Error: {str(e)}")