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
    system_prompt = (
        "You are a helpful assistant. Your purpose is to answer user's questions "
        "based on ONLY the provided context.\n"
        "If there is no answer for a question in context, say honestly: 'I don't know'. "
        "Do not imagine the answer.\n\n"
        "IMPORTANT: You MUST cite the source at the end of the sentence using the format [1], [2], etc., "
        "corresponding to the 'Document [X]' numbers provided in the context.\n\n"
        "Context:\n{context}"
    )

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
