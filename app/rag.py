import os

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

VECTOR_STORE_ID = os.environ["VECTOR_STORE_ID"]


SYSTEM_PROMPT = """
You are an enterprise knowledge assistant.

Answer questions using only information retrieved from the
company knowledge base.

If the knowledge base does not contain enough information
to answer the question, say that the information is not
available in the knowledge base.

Do not invent policies, numbers, dates, or procedures.

Give concise answers and identify the relevant source documents
when possible.
"""


def answer_question(question: str):
    response = client.responses.create(
        model="gpt-5-mini",
        instructions=SYSTEM_PROMPT,
        input=question,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [VECTOR_STORE_ID],
            }
        ],
    )

    return response