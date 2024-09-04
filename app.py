from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import Functions.document_llm_streamlit as dls
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📖",
)

st.title("DocumentGPT")

with st.sidebar:
    api_key = st.text_input("OpenAI API key를 입력해주세요.", type="password")
    if api_key:
        with st.spinner("API 키 확인 중..."):
            valid_key = dls.is_valid_openai_key(api_key)
            
        if valid_key:
            _llm = dls.get_llm(api_key)
            file = st.file_uploader(
                ".txt .pdf 혹은 .docx 유형의 파일을 업로드하세요.", type=["txt", "pdf", "docx"]
            )
        else:
            st.warning("API 키가 유효하지 않습니다. 다시 시도해주세요.")
    
    st.write("""
             ### Function.document_llm_streamlit.py ###
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
                    separator="\\n",
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
                return "\\n\\n".join(document.page_content for document in docs)

            prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        '''
                        당신은 문서를 정확하게 분석하는 전문가 입니다. 답변은 문서에 기반하여 해주세요. 모르는 내용은 모른다고 해주세요. 한국어로 답변해주세요.
                        
                        Context: context
                        ''',
                    ),
                    ("human", "question"),
                ])
    ### app.py ###
        st.set_page_config(
        page_title="DocumentGPT",
        page_icon="📖",
        )

        st.title("DocumentGPT")

        with st.sidebar:
            api_key = st.text_input("OpenAI API key를 입력해주세요.", type="password")
            if api_key:
                with st.spinner("API 키 확인 중..."):
                    valid_key = dls.is_valid_openai_key(api_key)
                    
                if valid_key:
                    _llm = dls.get_llm(api_key)
                    file = st.file_uploader(
                        ".txt .pdf 혹은 .docx 유형의 파일을 업로드하세요.", type=["txt", "pdf", "docx"]
                    )
                else:
                    st.warning("API 키가 유효하지 않습니다. 다시 시도해주세요.")
            
            if file:
            retriever = dls.embed_file(file)
            dls.send_message("준비완료! 무엇이든 물어보세요", "ai", save=False)
            dls.paint_history()
            message = st.chat_input("당신의 문서에 대해 무엇이든 물어보세요...")
            if message:
                dls.send_message(message, "human")
                chain = (
                    {
                        "context": retriever | RunnableLambda(dls.format_docs),
                        "question": RunnablePassthrough(),
                    }
                    | dls.prompt
                    | _llm
                )
                with st.chat_message("ai"):
                    response = chain.invoke(message)
        else:
            st.markdown(
                '''
                        환영합니다!
                        
                        이 챗봇을 사용하여 AI에게 문서 관련 질문을 해보세요!
                        
                        사이드 바에 문서를 업로드하면 시작 할 수 있습니다!
                        '''
            )
            st.session_state["messages"] = []
            
    ### git repository url ###
        https://github.com/skyGom/Fullstack_GPT_Document
                """)

if file:
    retriever = dls.embed_file(file)
    dls.send_message("준비완료! 무엇이든 물어보세요", "ai", save=False)
    dls.paint_history()
    message = st.chat_input("당신의 문서에 대해 무엇이든 물어보세요...")
    if message:
        dls.send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(dls.format_docs),
                "question": RunnablePassthrough(),
            }
            | dls.prompt
            | _llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.markdown(
        """
                환영합니다!
                
                이 챗봇을 사용하여 AI에게 문서 관련 질문을 해보세요!
                
                사이드 바에 문서를 업로드하면 시작 할 수 있습니다!
                """
    )
    st.session_state["messages"] = []
