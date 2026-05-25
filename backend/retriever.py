from vector_store import load_vector_store

# LOAD VECTOR STORE ONLY ONCE
vectorstore = load_vector_store()


def retrieve_documents(query, k=1):
    """
    Retrieve top-k most relevant chunks for a user query

    Args:
        query (str): User question
        k (int): Number of top documents to retrieve

    Returns:
        docs (list): Relevant document chunks
    """

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    docs = retriever.invoke(query)

    if not docs:
        raise ValueError("No relevant documents found.")

    return docs


if __name__ == "__main__":

    try:

        query = input("Ask your question: ")

        results = retrieve_documents(query)

        print(f"\nTop {len(results)} Retrieved Chunks:\n")

        for i, doc in enumerate(results, 1):

            print(f"--- Result {i} ---")

            print(doc.page_content[:700])

            print("\nMetadata:", doc.metadata)

            print("\n")

    except Exception as e:

        print(f"Error: {str(e)}")