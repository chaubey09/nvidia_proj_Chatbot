import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
import pickle
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image
import os
import fitz  # PyMuPDF for PDF parsing
import logging
import re

# Enable detailed logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Fetch the NVIDIA API key from st.secrets
try:
    nvidia_api_key = st.secrets["nvidia_api_key"]
    if not nvidia_api_key:
        raise ValueError("NVIDIA API Key not found! Make sure it's set in the secrets.toml file.")
except KeyError:
    st.error("NVIDIA API Key not found in secrets.toml. Please add it to use this app.")
    st.stop()

st.set_page_config(layout="wide")

# Directories for uploaded files and processed text files
DOCS_DIR = os.path.abspath("./uploaded_docs")
TEXT_DIR = os.path.abspath("./processed_texts")
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)


def clean_pdf_text(text):
    """Clean and structure extracted PDF text"""
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    text = re.sub(r'(o|)\s*', '• ', text)  # Replace bullet points
    text = re.sub(r'(\d+\.\s+[A-Z][a-z]+)', r'\n\1', text)  # Add line breaks for numbered lists
    text = re.sub(r'(\|\s*.+?\s*\|)', r'\n\1\n', text)  # Add line breaks for table-like structures
    text = re.sub(r'([a-zA-Z])\s*\n\s*([a-zA-Z])', r'\1 \2', text)  # Fix broken words across lines
    return text


# Sidebar for document upload and contact info
with st.sidebar:
    st.subheader("Add to the Knowledge Base")
    uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:", type=["txt", "pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DOCS_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            if uploaded_file.name.endswith(".pdf"):
                try:
                    extracted_text = fitz.open(file_path)
                    text = "".join([page.get_text("text") for page in extracted_text if page.get_text("text").strip()])
                    if text.strip():
                        cleaned_text = clean_pdf_text(text)
                        txt_filename = os.path.join(TEXT_DIR, uploaded_file.name.replace(".pdf", ".txt"))
                        with open(txt_filename, "w", encoding="utf-8") as f:
                            f.write(cleaned_text)
                    else:
                        st.warning(f"No extractable text found in {uploaded_file.name}. Skipping.")
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")
                    logging.error(f"Error processing PDF {uploaded_file.name}: {e}")

    uploaded_files_list = os.listdir(DOCS_DIR)
    if uploaded_files_list:
        st.subheader("Current Documents")
        for doc in uploaded_files_list:
            st.write(f"📄 {doc}")
            if st.button(f"Delete {doc}"):
                os.remove(os.path.join(DOCS_DIR, doc))
                st.success(f"Deleted {doc}")
                txt_filename = os.path.join(TEXT_DIR, doc.replace(".pdf", ".txt"))
                if os.path.exists(txt_filename):
                    os.remove(txt_filename)
                    st.success(f"Deleted processed text file for {doc}")
                vector_store_path = "vectorstore.pkl"
                if os.path.exists(vector_store_path):
                    os.remove(vector_store_path)
                raw_documents = DirectoryLoader(TEXT_DIR, glob="*.txt").load()
                if raw_documents:
                    try:
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=300,
                            chunk_overlap=50,
                            separators=["\n\n", "\n", " ", ""]
                        )
                        documents = text_splitter.split_documents(raw_documents)
                        vectorstore = FAISS.from_documents(documents, document_embedder)
                        with open(vector_store_path, "wb") as f:
                            pickle.dump(vectorstore, f)
                        st.success("Vector store updated successfully.")
                    except Exception as e:
                        st.error(f"Failed to rebuild vector store: {e}")
                st.rerun()

st.sidebar.subheader("Contact Information")
try:
    profile_pic = Image.open("profile_photo.png")
    st.sidebar.image(profile_pic, width=150, use_container_width=False, caption="Anmol Chaubey", output_format="PNG")
except FileNotFoundError:
    st.sidebar.warning("Profile photo not found. Skipping image display.")

st.sidebar.markdown("""
    **Name:** Anmol Chaubey  
    **Email:** anmolchaubey820@gmail.com  
    [LinkedIn](https://www.linkedin.com/in/anmol-chaubey-120b42206/)
""")

assistant_name = "AskAI"
personality = st.sidebar.radio("Choose Assistant Personality", ["Formal", "Casual", "Humorous"], index=1)

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.write("Chat cleared.")

# Embedding Model and LLM
try:
    llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", max_tokens=300, temperature=0.2, api_key=nvidia_api_key)
    document_embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", model_type="passage", api_key=nvidia_api_key)
except Exception as e:
    st.error(f"Failed to initialize NVIDIA services: {e}")
    st.stop()

# Vector Database Store
vector_store_path = "vectorstore.pkl"
raw_documents = DirectoryLoader(TEXT_DIR, glob="*.txt").load()
vector_store_exists = os.path.exists(vector_store_path)
vectorstore = None

if vector_store_exists:
    try:
        with open(vector_store_path, "rb") as f:
            vectorstore = pickle.load(f)
        st.sidebar.success("Existing vector store loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        vectorstore = None
elif raw_documents:
    with st.spinner("Processing documents..."):
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50,
                separators=["\n\n", "\n", " ", ""]
            )
            documents = text_splitter.split_documents(raw_documents)
            vectorstore = FAISS.from_documents(documents, document_embedder)
            with open(vector_store_path, "wb") as f:
                pickle.dump(vectorstore, f)
            st.success("Vector store created and saved.")
        except Exception as e:
            st.error(f"Failed to process documents: {e}")
else:
    st.sidebar.warning("No documents available to process!", icon="⚠️")

# Chat Interface
st.subheader(f"Chat with {assistant_name} ({personality} Mode)")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Updated prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", f"""You are a helpful AI assistant named {assistant_name}. You communicate in a {personality.lower()} tone.
If the user's query is a specific question, provide a concise and direct answer first, citing the relevant document excerpt. Then, if relevant, offer additional context or a summary.
If the query asks for a general summary, provide a structured summary with these sections:
1. Document Overview
2. Key Topics
3. Important Comparisons (if any)
4. Use Cases/Examples
5. Technical Specifications (if relevant)
Format responses with clear headings and bullet points for readability.
Document Context: {{context}}"""),
    ("user", "{{input}}")
])


def extract_birth_date(context, query):
    """Extract birth date from context if query is about birth"""
    if "born" in query.lower() and ("choubey" in query.lower() or "mr." in query.lower() or "rajiv" in query.lower()):
        match = re.search(r'(?:born|b\W*orn)\W*(\d{1,2}(?:st|nd|rd|th)?\s+\w+,\s+\d{4})', context, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def clean_response(response):
    """Clean and format LLM response"""
    response = re.sub(r'(^|\n)\s*[•o]\s*', '\n• ', response)
    response = re.sub(r'\n{3,}', '\n\n', response)
    response = re.sub(r'(\d+)\.\s+', r'\1. ', response)
    # Remove generic placeholders
    response = re.sub(r'(?i)let me check that for you|haven\'t asked a question', '', response)
    return response.strip()


# Chat input
user_input = st.chat_input("Enter your prompt here:")
if user_input:
    logging.debug(f"User Input: {user_input}")
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if vectorstore:
        try:
            relevant_docs = vectorstore.similarity_search(user_input, k=5)  # Increased k for better coverage
            context = "\n".join([f"**Document Excerpt {i+1}:**\n{doc.page_content.strip()}" 
                                 for i, doc in enumerate(relevant_docs) if doc.page_content.strip()])
            logging.debug(f"Retrieved Context: {context}")
            # Display context for debugging
            with st.expander("Debug: Retrieved Context"):
                st.write(context if context else "No context retrieved.")
        except Exception as e:
            st.error(f"Error retrieving context: {e}")
            context = ""
    else:
        context = ""
        st.warning("No vector store available. Please upload documents.")

    # Extract specific answer for birth date
    birth_date = extract_birth_date(context, user_input)
    if birth_date:
        direct_answer = f"Mr. Choubey was born on {birth_date}.\n**Additional Context**:\n"
    else:
        direct_answer = ""

    # Format the prompt
    prompt = prompt_template.format_messages(context=context, input=user_input)

    # Invoke the LLM
    try:
        response = llm.invoke(prompt).content
        cleaned_response = direct_answer + clean_response(response)
        if not direct_answer and not response.strip():
            cleaned_response = "I couldn't find an answer in the documents. Try uploading more files or rephrasing your question."
    except Exception as e:
        st.error(f"Error generating response: {e}")
        cleaned_response = "Sorry, I encountered an error while generating a response."

    st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
    with st.chat_message("assistant"):
        st.markdown(cleaned_response)
