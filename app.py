import streamlit as st
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai_embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

openai_api_key = st.secrets["OPENAI_API_KEY"]

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

def read_pdf(pdf_paths):
    text = ""
    for path in pdf_paths:
        pdf_reader = PdfReader(path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")
    return vector_store

def get_retriever_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0)

    system_message_prompt = SystemMessagePromptTemplate(
        system_message="""You are a helpful AI assistant. You will answer questions using information from the provided PDF documents. 
        If the answer is not available in the documents, you will politely indicate that the answer is not available."""
    )

    human_prompt = HumanMessagePromptTemplate(human_message="{question}")

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_prompt])

    retriever_chain = RetrievalQAWithSourcesChain.from_llm(llm, vectorstore=vector_store, qa_prompt=chat_prompt)
    return retriever_chain

def main():
    st.set_page_config("PDF Chat")
    st.title("PDF Chat")

    with st.sidebar:
        st.title("Menu")
        pdf_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                text = read_pdf(pdf_files)
                text_chunks = get_chunks(text)
                vector_store = create_vector_store(text_chunks)
                st.success("PDFs processed successfully!")

    user_question = st.text_input("Ask a question about the PDF content")

    if user_question:
        retriever_chain = get_retriever_chain(vector_store)
        result = retriever_chain({"question": user_question})
        st.write("Answer:", result["result"])
        st.write("Sources:", result["sources"])

if __name__ == "__main__":
    main()