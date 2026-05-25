from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

import os


def load_documents(file_path):
    """
    Load PDF or TXT files

    Args:
        file_path (str): Path to file

    Returns:
        documents (list): List of LangChain documents
    """

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found at path: {file_path}"
        )

    # Detect file extension
    extension = os.path.splitext(file_path)[1].lower()

    # PDF Loader
    if extension == ".pdf":

        loader = PyPDFLoader(file_path)

    # TXT Loader
    elif extension == ".txt":

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

    else:

        raise ValueError(
            "Unsupported file type. Use PDF or TXT."
        )

    # Load documents
    documents = loader.load()

    # Validate extraction
    if not documents:

        raise ValueError(
            "No content found inside the file."
        )

    return documents


if __name__ == "__main__":

    file_path = "data/sample.txt"   # Change to your file

    try:

        docs = load_documents(file_path)

        print(f"Total pages/chunks loaded: {len(docs)}")

        print("\nFirst document preview:\n")

        print(docs[0].page_content[:1000])

    except Exception as e:

        print(f"Error: {str(e)}")