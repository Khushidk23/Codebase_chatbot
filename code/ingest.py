from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS

def create_vectorstore():
    loader = DirectoryLoader("./code", glob="**/*.py", loader_cls=TextLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local("vectorstore")

if __name__ == "__main__":
    create_vectorstore()
