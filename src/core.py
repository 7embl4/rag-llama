from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableSerializable

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


def setup_model(
    collection_name: str,
    db_dir: str = "./db",
    k_docs: int = 3,
    llm_model: str = "llama3.1:8b",
    embeddings_model: str = "nomic-embed-text"
) -> tuple[RunnableSerializable, VectorStoreRetriever]:
    """
    Setuping retriever and chain for LLM

    Args:
        collection_name (str): name of a collection in vector database
        db_dir (str): directory of a vector database
        k_docs (int): top k documents from database
        llm_model (str): name of a LLM
        embeddings_model (str): name of an embedding model
    """
    # database
    embeddings = OllamaEmbeddings(model=embeddings_model)
    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=db_dir,
        embedding_function=embeddings,
    )
    retriever = vector_store.as_retriever(
        search_kwargs={"k": k_docs}
    )

    # model
    model = OllamaLLM(model=llm_model)

    # prompt
    system_prompt = """
        You are a helpful assistant. Your purpose is to answer user's questions 
        based on ONLY the provided context.\n
        Carefully analyze the context. If the context contains related information or formulas, 
        synthesize the answer from those pieces. Do not refuse to answer if the exact word-for-word 
        definition is missing, but try to construct a meaningful answer from what is provided.\n
        If the context is absolutely irrelevant to the question, say 'I don't know'.\n\n
        Do NOT use any citations when saying 'I don't know'.\n\n
        INSTRUCTIONS FOR CITATIONS:\n
        When you state a fact from the context, you MUST append the citation number 
        at the end of the sentence in the format [1], [2], etc., matching the 'Document [X]' numbers. 
        Do not cite bibliographies or reference lists unless directly asked.\n\n
        Context:\n{context}
    """

    # template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # chain
    chain = prompt | model | StrOutputParser()
    return chain, retriever


def ask_question(
    user_question: str,
    chain: RunnableSerializable,
    retriever: VectorStoreRetriever,
    chat_history: list
) -> tuple[str, list[Document]]:
    """
    Asking question to LLM via chain

    Args: 
        user_question (str): user question to LLM
        chain (RunnableSerializable): chain of a LLM
        retriever (VectorStoreRetriever): retriever of a vector database
    """
    # find relevant chunks
    print("Looking in database...")
    relevant_docs = retriever.invoke(user_question)

    # put all the pieces together
    context_parts = []
    for i, doc in enumerate(relevant_docs):
        context_parts.append(f"Document [{i+1}]:\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    # get the answer
    print("Generating answer...")
    answer = chain.invoke({
        "context": context,
        "input": user_question,
        "chat_history": chat_history
    })

    return answer, relevant_docs
