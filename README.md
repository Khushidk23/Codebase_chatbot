
# 💻 Codebase Chatbot

A simple chatbot interface built using Streamlit and LangChain to query your codebase intelligently using local embeddings and the **LLaMA 3 model** via **Ollama**.

---

## 🚀 Features

* 🔍 Ask questions about your codebase.
* 🤖 Powered by **LLaMA 3** via Ollama for natural language understanding.
* 📚 Retrieves relevant code snippets using FAISS vector search.
* 🧠 Custom text chunking logic using LangChain's `RecursiveCharacterTextSplitter`.
* 📦 Local embeddings — no need to send data to external APIs.

---

## 🧰 Tech Stack

* **Frontend**: Streamlit
* **Backend**: Python, LangChain
* **Embeddings & LLM**: Ollama (`llama3`)
* **Vector Store**: FAISS
* **Text Splitters**: LangChain `RecursiveCharacterTextSplitter` and `CharacterTextSplitter`

---



## 🛠️ Setup Instructions

### 1. Install Dependencies

Make sure you have Python 3.8 or above.
Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Install Ollama

Make sure [Ollama](https://ollama.com) is installed and running locally.
Then pull the `llama3` model:

```bash
ollama pull llama3
```

### 3. Prepare Your Vector Store

You need to embed your codebase before chatting with it. Create a script like this:

```python
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and chunk documents
loader = DirectoryLoader("codebase", glob="**/*.py", show_progress=True)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embed and store
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("vectorstore")
```

Run this script once to generate the vectorstore.

### 4. Launch the App

```bash
streamlit run app.py
```

---

## 📸 Screenshot

![Chat UI](https://via.placeholder.com/800x400?text=Codebase+Chatbot+UI)

---

## ❓How it Works

* Takes user input via Streamlit text input.
* Loads the prebuilt FAISS index.
* Retrieves relevant chunks using cosine similarity.
* Passes context + query to `llama3` via LangChain’s `RetrievalQA` chain.
* Displays the final answer.

---

## 🧪 Example Queries

* "What does the `split_text` function do?"
* "Where are chunking strategies implemented?"
* "How is the FAISS index built?"

---

## 📌 Requirements

* Python 3.8+
* Ollama installed and running
* FAISS
* LangChain (`langchain`, `langchain-community`, and optionally `langchain-openai`)
* Streamlit

---

## 🔒 Privacy

All code and embeddings are processed **locally** — nothing is sent to the cloud.

---

## 📬 Contact

Feel free to reach out if you have questions or suggestions!
GitHub: [@Khushidk23](https://github.com/Khushidk23)

---
