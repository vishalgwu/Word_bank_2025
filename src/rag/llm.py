import os
from openai import OpenAI


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Please add it to your .env file."
        )

    return OpenAI(api_key=api_key)


def call_llm(prompt: str) -> str:
    """
    Calls OpenAI safely.
    """

    client = get_client()  # create client AFTER env loads

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content


def generate_answer(query: str, retrieved_chunks: list[str]) -> str:
    """
    Final answer builder using retrieved context + LLM.
    """

    context_text = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a data analysis assistant for a global labor dataset.
Use ONLY the context below to answer the question.
If the context does not contain the answer, say 'The dataset does not include this information.'

Context:
{context_text}

Question: {query}

Answer concisely:
"""

    return call_llm(prompt)
