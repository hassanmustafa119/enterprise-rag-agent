from langchain_groq import ChatGroq
from retriever import retrieve_documents
from dotenv import load_dotenv
import os

load_dotenv()


def get_llm():
    """
    Initialize Groq LLM
    """

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=400,
        top_p=0.7
    )

    return llm


def generate_answer(query):
    """
    Retrieve context + generate final answer
    """

    # Retrieve relevant chunks
    docs = retrieve_documents(query)

    # Combine retrieved context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Prompt to LLM
    prompt = f"""
You are an advanced enterprise AI assistant.

Answer the user's question using only the provided context.

Guidelines:
- Give clear, natural, and human-friendly answers.
- Keep responses balanced: not too short and not too detailed.
- Answer according to the user's query complexity.
- Use simple and easy-to-understand language.
- Avoid unnecessary explanations, repetition, or extra details.
- Use short paragraphs or bullet points when needed.
- If the answer is not available in the context, say:
  "I could not find this information in the document."

Context:
{context}

Question:
{query}

Answer:
"""

    # Initialize model
    llm = get_llm()

    # Generate response
    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":

    try:

        query = input("Ask your question: ")

        answer = generate_answer(query)

        print("\nFinal AI Answer:\n")
        print(answer)

    except Exception as e:

        print(f"Error: {str(e)}")