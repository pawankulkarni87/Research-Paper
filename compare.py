from langchain.chat_models import ChatOpenAI

def compare_docs(text1, text2, focus="methodology"):
    llm = ChatOpenAI(model="gpt-4o",temperature=0)
    prompt = f"""Compare these two papers focusing on {focus}:
    --- Paper A ---
    {text1}
    --- Paper B ---
    {text2}
    """
    return llm.predict(prompt)

