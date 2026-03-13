def input_processing(state):

    query = state["query"].strip()

    return {
        **state,
        "query": query
    }


# def retrieval_node(state):
#
#     query = state.get("query")
#     vector_db = state.get("vector_db")
#
#     if vector_db is None:
#         return {**state, "docs": []}
#
#     docs = vector_db.similarity_search(query, k=3)
#
#     return {
#         **state,
#         "docs": docs
#     }
def retrieval_node(state):

    query = state.get("query")
    vector_db = state.get("vector_db")

    docs = vector_db.similarity_search(query, k=3)

    print("Retrieved docs:", docs)

    return {
        **state,
        "docs": docs
    }


def context_validation(state):

    docs = state.get("docs", [])

    valid = len(docs) > 0

    return {
        **state,
        "valid": valid
    }


def generation_node(state):

    query = state.get("query")
    docs = state.get("docs", [])
    llm = state.get("llm")

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Use the context to answer the question.
If answer not found say I don't know.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "response": response
    }


def fallback_node(state):

    return {
        **state,
        "response": "Sorry, I couldn't find relevant information."
    }


def formatter_node(state):

    response = state.get("response")

    return {
        **state,
        "final_response": response
    }