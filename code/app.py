# app.py
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

st.title("💻 Codebase Chatbot")

query = st.text_input("Ask a question about the codebase...")

if query:
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = FAISS.load_local("vectorstore", embeddings)
    retriever = vectorstore.as_retriever()

    llm = Ollama(model="llama3")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = qa.run(query)

    st.write("**Answer:**", answer)
