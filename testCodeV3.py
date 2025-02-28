import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,HumanMessage,trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory,RunnablePassthrough
from operator import itemgetter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain.chains.summarize import load_summarize_chain

load_dotenv()


groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


#create embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## set up Streamlit 
st.title("Conversational ChatBot For PDF")
st.write("Upload Pdf's and chat with their content")

option = st.selectbox(
    "Choose LLM Model:",
    ["llama-3.3-70b-versatile", "qwen-2.5-32b", "gemma2-9b-it", "mistral-saba-24b"]
)
if(option):
    #initialize model
    llm = ChatGroq(model=option,groq_api_key=groq_api_key)

    if 'store' not in st.session_state:
        st.session_state.store={}

    session_id = st.text_input("Session ID",value="default_session")
    uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
    col1, col2 = st.columns(2)
    ## Process uploaded  PDF's
    documents=[]
    if uploaded_files:
        for idx, uploaded_file in enumerate(uploaded_files):
            temppdf=f"./temp_{idx}.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
            os.remove(temppdf)



        #create vector database and retriver by loading the pdf

        text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
        splits = text_spliter.split_documents(documents)
        chromadb = Chroma.from_documents(documents=splits,embedding=embeddings,persist_directory="db")
        retriever = chromadb.as_retriever(search_type="mmr",kwargs={"k":3})

        ## Prompt Template
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        trimmer = trim_messages(
            max_tokens = 300,
            strategy = "last",
            token_counter = llm,
            include_system = True,
            start_on = "human"
        )

        parser = StrOutputParser()

        question_answering_chain =  (
            RunnablePassthrough.assign(chat_history=itemgetter("chat_history")|trimmer)
            | qa_prompt
            | llm
            | parser
        )

        rag_chain = create_retrieval_chain(history_aware_retriever,question_answering_chain)


        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if(session_id not in st.session_state.store):
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        summarize_chain = load_summarize_chain(llm=llm,chain_type="refine")
        user_input = st.text_input("Your question:")
        with col1:
            button1 = st.button('Ask Question')

        with col2:
            button2 = st.button('Summarize')
        if(button1):
            
            if user_input:
                session_history=get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id":session_id}
                    }
                )
            
                st.write("Assistant:", response['answer'])

        elif(button2):
            results=summarize_chain.invoke(documents)
            st.write("Summary")
            st.write(results["output_text"])

        if 'store' in st.session_state and session_id in st.session_state.store:
            with st.expander("📜 Chat History", expanded=False): 
                session_history = st.session_state.store[session_id]
                
                for msg in session_history.messages:
                    if isinstance(msg, HumanMessage):
                        st.markdown(f"**🧑‍💻 You:** {msg.content}")
                    else:
                        st.markdown(f"**🤖 Assistant:** {msg.content}")
        else:
            st.write("No chat history for this session yet.")
