import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import fitz  # PyMuPDF for PDF parsing
from PIL import Image
import os
import pickle
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Streamlit page
st.set_page_config(page_title="Document Q&A Chatbot", layout="wide")
st.title("Document Q&A Chatbot")

# Directories for uploaded and processed files
DOCS_DIR = os.path.abspath("./uploaded_docs")
TEXT_DIR = os.path.abspath("./processed_texts")
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# Function to clean extracted PDF text
def clean_pdf_text(text):
    """Clean and normalize extracted PDF text."""
    if not text:
        return ""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text).strip()      # Normalize whitespace
    text = re.sub(r'(\d+\.\s+[A-Z][a-z]+)', r'\n\1', text)  # Add line breaks for numbered lists
    text = re.sub(r'([a-zA-Z])\s*\n\s*([a-zA-Z])', r'\1 \2', text)  # Fix broken words
    return text

# Initialize Groq LLM
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        max_tokens=300,
        temperature=0.2
    )
except Exception as e:
    st.error(f"Failed to initialize Groq LLM: {str(e)}")
    logging.error(f"Error initializing Groq LLM: {str(e)}")
    st.stop()

# HuggingFace embeddings — runs locally, no API key needed
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

document_embedder = load_embeddings()

# Load or create vector store
vector_store_path = "vectorstore.pkl"
vectorstore = None

if os.path.exists(vector_store_path):
    try:
        with open(vector_store_path, "rb") as f:
            vectorstore = pickle.load(f)
        st.sidebar.success("Loaded existing vector store.")
    except Exception as e:
        st.sidebar.error(f"Failed to load vector store: {str(e)}")
        logging.error(f"Error loading vector store: {str(e)}")
        os.remove(vector_store_path)

if not vectorstore:
    raw_documents = DirectoryLoader(TEXT_DIR, glob="*.txt", loader_cls=TextLoader).load()
    if raw_documents:
        try:
            with st.spinner("Processing documents..."):
                st.sidebar.info(f"Processing files: {[os.path.basename(doc.metadata['source']) for doc in raw_documents]}")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                documents = text_splitter.split_documents(raw_documents)
                vectorstore = FAISS.from_documents(documents, document_embedder)
                with open(vector_store_path, "wb") as f:
                    pickle.dump(vectorstore, f)
                st.success("Vector store created and saved.")
        except Exception as e:
            st.error(f"Failed to create vector store: {str(e)}")
            logging.error(f"Error creating vector store: {str(e)}")
    else:
        st.sidebar.warning("No documents available to process. Please upload files.")

# Sidebar for file upload, management, personality selection, and credentials
with st.sidebar:
    st.subheader("Manage Documents")
    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                file_path = os.path.join(DOCS_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                if uploaded_file.name.endswith(".pdf"):
                    pdf_document = fitz.open(file_path)
                    text = "".join([page.get_text("text") for page in pdf_document if page.get_text("text").strip()])
                    pdf_document.close()
                    if text:
                        cleaned_text = clean_pdf_text(text)
                        txt_path = os.path.join(TEXT_DIR, uploaded_file.name.replace(".pdf", ".txt"))
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(cleaned_text)
                        st.success(f"Processed {uploaded_file.name} to text.")
                    else:
                        st.warning(f"No text extracted from {uploaded_file.name}.")
                elif uploaded_file.name.endswith(".txt"):
                    txt_path = os.path.join(TEXT_DIR, uploaded_file.name)
                    with open(txt_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Uploaded {uploaded_file.name}.")
            except Exception as e:
                st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
                logging.error(f"Error processing {uploaded_file.name}: {str(e)}")

        # Rebuild vector store after upload
        if os.path.exists(vector_store_path):
            os.remove(vector_store_path)
        raw_documents = DirectoryLoader(TEXT_DIR, glob="*.txt", loader_cls=TextLoader).load()
        if raw_documents:
            try:
                st.sidebar.info(f"Processing files: {[os.path.basename(doc.metadata['source']) for doc in raw_documents]}")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                documents = text_splitter.split_documents(raw_documents)
                vectorstore = FAISS.from_documents(documents, document_embedder)
                with open(vector_store_path, "wb") as f:
                    pickle.dump(vectorstore, f)
                st.success("Vector store updated.")
            except Exception as e:
                st.error(f"Failed to rebuild vector store: {str(e)}")
                logging.error(f"Error rebuilding vector store: {str(e)}")

    # Display and delete documents
    st.subheader("Current Documents")
    for doc in os.listdir(DOCS_DIR):
        st.write(f"📄 {doc}")
        if st.button(f"Delete {doc}", key=f"delete_{doc}"):
            try:
                os.remove(os.path.join(DOCS_DIR, doc))
                txt_path = os.path.join(TEXT_DIR, doc.replace(".pdf", ".txt"))
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                if os.path.exists(vector_store_path):
                    os.remove(vector_store_path)
                raw_documents = DirectoryLoader(TEXT_DIR, glob="*.txt", loader_cls=TextLoader).load()
                if raw_documents:
                    st.sidebar.info(f"Processing files: {[os.path.basename(doc.metadata['source']) for doc in raw_documents]}")
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                    documents = text_splitter.split_documents(raw_documents)
                    vectorstore = FAISS.from_documents(documents, document_embedder)
                    with open(vector_store_path, "wb") as f:
                        pickle.dump(vectorstore, f)
                    st.success("Vector store updated.")
                else:
                    vectorstore = None
                    st.info("No documents left.")
                st.success(f"Deleted {doc}.")
            except Exception as e:
                st.error(f"Failed to delete {doc}: {str(e)}")
                logging.error(f"Error deleting {doc}: {str(e)}")

    # Personality selection
    st.subheader("Assistant Personality")
    personality = st.radio("Choose tone:", ["Formal", "Casual", "Humorous"], index=1)

    # Credentials
    st.subheader("Contact Information")
    profile_pic = Image.open("profile_photo.png")
    st.image(profile_pic, width=150, caption="Anmol Chaubey", output_format="PNG")
    st.markdown("""
    **Name:** Anmol Chaubey  
    **Email:** anmolchaubey820@gmail.com  
    [LinkedIn](https://www.linkedin.com/in/anmol-chaubey-120b42206/)
    """)

# Chat interface
st.subheader(f"Chat with Documents ({personality} Mode)")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Prompt template with personality modes
prompt_template = ChatPromptTemplate.from_template("""
You are a document-based Q&A assistant created by Anmol Chaubey. Communicate in a {personality} tone:
- Formal: Use professional, concise, and academic language.
- Casual: Use friendly, conversational language, like explaining to a peer.
- Humorous: Use lighthearted, playful language with witty remarks, but stay accurate.
Answer the question only if relevant information is found in the provided context. If the context is unrelated to the question or no relevant information is found, respond with: "No relevant information found in the documents. Please ask a question related to the uploaded documents."

Question: {question}

Context: {context}

Answer:
""")

# Chat input handling
if vectorstore:
    user_input = st.chat_input("Ask a question about the documents:")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            # Retrieve context
            docs = vectorstore.similarity_search(user_input, k=4)
            context = "\n".join([doc.page_content for doc in docs])

            # Check context relevance
            if not any(word.lower() in context.lower() for word in user_input.lower().split()):
                response = "No relevant information found in the documents. Please ask a question related to the uploaded documents."
            else:
                # Generate response
                with st.spinner("Generating response..."):
                    response = llm.invoke(prompt_template.format_messages(
                        personality=personality.lower(),
                        question=user_input,
                        context=context
                    )).content
                    if not response.strip() or "no relevant information" in response.lower():
                        response = "No relevant information found in the documents. Please ask a question related to the uploaded documents."

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

            # Debug context
            with st.expander("Debug: Retrieved Context"):
                for i, doc in enumerate(docs):
                    st.write(f"Excerpt {i+1}: {doc.page_content}")
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            logging.error(f"Error generating response: {str(e)}")
else:
    st.warning("No documents uploaded. Please add files to start chatting.")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.success("Chat cleared.")

