#https://www.youtube.com/watch?v=oi5Jwrc_ze4
#python3.11 -m streamlit run Langchain_Chat_App.py
#https://blog.langchain.dev/chat-your-data-submissions/
#https://blog.streamlit.io/ai-talks-chatgpt-assistant-via-streamlit/
#https://www.pinecone.io/learn/langchain-conversational-memory/

#pip3 install Langchain streamlit streamlit_chat openaifaiss-cpu

import os

#To load the Vector Store from files:
# Upload the files `$DATA_STORE_DIR/index.faiss` and `$DATA_STORE_DIR/index.pkl` to local
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationChain

import streamlit as st
from streamlit_chat import message
import random
import time

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token + ""
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(self.text)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")



if os.path.exists('DATA_STORE_DIR_v2'):
  vector_store = FAISS.load_local(
      'DATA_STORE_DIR_v2',
      OpenAIEmbeddings()
  )
else:
  print(f"Missing files. Upload index.faiss and index.pkl files to {'DATA_STORE_DIR'} directory first")


#Query using the vector store with ChatGPT integration

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

system_template="""You are a helpful sales agent working for Unique Japan Tours. You have been trained on all the information on the company website.
Use the following pieces of context to answer the users question. Take note of the sources and include them at the end of your reply, use "SOURCES:" in capital letters regardless of the number of sources, followed by the url hyperlink of each source.
If you don't know the answer, just say that "I don't know", don't try to make up an answer but refer this query to the team at Unique Japan Tours who can help (https://www.uniquejapantours.com/contact-us/)
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

chain_type_kwargs = {"prompt": prompt}

# Streamlit emoji's - https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
# Setting page title and header
st.set_page_config(page_title="Chatbot", page_icon=":mount_fuji:")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == os.environ["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

# How to add password - https://docs.streamlit.io/knowledge-base/deploy/authentication-without-sso
if check_password():

    #Create the Chatbot Interface

    st.title("Japan Travel Planner")

    # https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
    with st.chat_message("assistant"):
        st.write("Hello 👋. ")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])




    # React to user input
    if query := st.chat_input("Where you want to go, what types of activities, with who and for how long ?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        # Display assistant response in chat message container
        with (st.chat_message("assistant")):
            message_placeholder = st.empty()
            full_response = ""

            #Define & call model
            stream_handler = StreamHandler(message_placeholder, display_method='write')
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, callbacks=[stream_handler], temperature=0,
                             max_tokens=256)  # Modify model_name if you have access to GPT-4
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(),
                return_source_documents=True,
                chain_type_kwargs=chain_type_kwargs,
                verbose=True
            )

            response = chain({'question': query})
            full_response = response.get("answer", "")
            #message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

