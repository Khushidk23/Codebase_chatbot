import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load OpenAI API key from .env
load_dotenv()

UPLOAD_DIR = "docs"
FAISS_INDEX_DIR = "faiss_index"

st.set_page_config(page_title="💬 Ask Questions on Your Documents")
st.title("💬 Ask Questions on Your Documents")

# Step 1: Upload PDFs
uploaded_files = st.file_uploader("📤 Upload 1 or more PDF files", type="pdf", accept_multiple_files=True)

# Step 2: Choose LLM model
model_name = st.selectbox("🧠 Choose a model", ["gpt-3.5-turbo", "gpt-4"])

# Step 3: Save uploaded files
def save_uploaded_files(files):
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    for file in files:
        with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
            f.write(file.getbuffer())
    st.success(f"✅ Uploaded {len(files)} file(s) successfully!")

# Step 4: Build FAISS index
def build_faiss_index():
    loader = DirectoryLoader(UPLOAD_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_INDEX_DIR)
    return db

# Step 5: Load FAISS index
def load_faiss_index():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(FAISS_INDEX_DIR, embeddings)

# Step 6: Answer questions
def ask_question(question, model_name):
    vectordb = load_faiss_index()
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(question)

# Step 7: Handle upload and build index
if uploaded_files:
    save_uploaded_files(uploaded_files)
    build_faiss_index()
    st.success("✅ Embeddings created. You can now ask questions!")

# Step 8: Ask question
if os.path.exists(FAISS_INDEX_DIR):
    user_question = st.text_input("💬 Ask a question based on your data:")
    
    if user_question:
        answer = ask_question(user_question, model_name)
        st.markdown(f"**Answer:** {answer}")
