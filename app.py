import streamlit as st

from src.utils import load_docs
from src.core import setup_model, ask_question

from langchain_core.messages import HumanMessage, AIMessage


DATA_DIR = "data"
COLLECTION_NAME = "files"
st.title("kinda helper")

# load and setup
load_docs(
    data_dir=DATA_DIR,
    collection_name=COLLECTION_NAME
)
chain, retriever = setup_model(
    collection_name=COLLECTION_NAME
)

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# show history
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# response to user
user_input = st.chat_input()
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append(
        HumanMessage(content=user_input)
    )

    response, relevant_docs = ask_question(
        user_input, chain, retriever, st.session_state.messages
    )
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append(
        AIMessage(content=response)
    )

    if relevant_docs:
        with st.expander("Sources"):
            for i, doc in enumerate(relevant_docs):
                source_name = doc.metadata.get("source", "Unknown").split("/")[-1]
                page_num = doc.metadata.get("page", "Unknown")
                
                st.markdown(f"**Source [{i+1}]: {source_name} (Page {page_num})**")
                
                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                st.divider()
    