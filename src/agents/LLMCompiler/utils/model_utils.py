from langchain_openai import ChatOpenAI

def get_model(
    model_name,
    temperature=0,
    host=None,
    port=None,
    max_tokens=None,
):
    if host:
        base_url = f"http://{host}:{port}/v1"
    else:
        base_url = None

    llm = ChatOpenAI(
        base_url=base_url,
        model=model_name,
        temperature=temperature,
        max_retries=1,
        max_tokens=max_tokens,
    )

    return llm
