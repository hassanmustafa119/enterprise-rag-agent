from vector_store import load_vector_store

vectorstore = None


def get_vectorstore():
    """
    Lazy-load FAISS only when needed (Streamlit-safe)
    """
    global vectorstore

    if vectorstore is None:
        vectorstore = load_vector_store()

    return vectorstore


def retrieve_documents(query, k=1):
    vs = get_vectorstore()

    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    return retriever.invoke(query)


# CLI TEST
if __name__ == "__main__":

    try:
        query = input("Ask your question: ")
        results = retrieve_documents(query)

        print(f"\nTop {len(results)} Results:\n")

        for i, doc in enumerate(results, 1):
            print(f"--- Result {i} ---")
            print(doc.page_content[:700])
            print("\nMetadata:", doc.metadata)
            print("\n")

    except Exception as e:
        print(f"Error: {e}")