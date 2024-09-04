from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import openai
import openai.error
import streamlit as st

class ChatCallBackHandler(BaseCallbackHandler):
    
    message = ""
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
            
    def on_llm_end(self, *arg, **kwargs):
        save_message(self.message, "ai")
            
    def on_llm_new_token(self, token, *arg, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)
        
key = None

def is_valid_openai_key(key):
    try:
        openai.api_key = key
        openai.Model.list()
        return True
    except Exception as e:
        return False
    
def get_llm(api_key):
    key = api_key
    return ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.1,
            streaming=True,
            callbacks=[ChatCallBackHandler()],
            openai_api_key=key,
        )
    
@st.cache_data(show_spinner="파일을 임베딩 하고있습니다...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            당신은 문서를 정확하게 분석하는 전문가 입니다. 답변은 문서에 기반하여 해주세요. 모르는 내용은 모른다고 해주세요. 한국어로 답변해주세요.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ])