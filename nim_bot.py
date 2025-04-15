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

# Sidebar for document upload and contact info
with st.sidebar:
    st.subheader("Add to the Knowledge Base")

    DOCS_DIR = os.path.abspath("./uploaded_docs")
    os.makedirs(DOCS_DIR, exist_ok=True)

    uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:", type=["txt", "pdf"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DOCS_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"File {uploaded_file.name} uploaded successfully!")

            # If it's a PDF, extract text and save as .txt
            if uploaded_file.name.endswith(".pdf"):
                try:
                    extracted_text = fitz.open(file_path)
                    text = "".join([page.get_text() for page in extracted_text if page.get_text().strip()])
                    if text.strip():  # Ensure the text is not empty
                        txt_filename = file_path.replace(".pdf", ".txt")
                        with open(txt_filename, "w") as f:
                            f.write(text)
                    else:
                        st.warning(f"No extractable text found in {uploaded_file.name}. Skipping.")
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")
                    logging.error(f"Error processing PDF {uploaded_file.name}: {e}")

    # Show preview of uploaded files
    if os.listdir(DOCS_DIR):
        st.subheader("Current Documents")
        for doc in os.listdir(DOCS_DIR):
            st.write(f"📄 {doc}")
            if st.button(f"Delete {doc}"):
                os.remove(os.path.join(DOCS_DIR, doc))
                st.success(f"Deleted {doc}")

st.sidebar.subheader("Contact Information")
profile_pic = Image.open("profile_photo.png")
st.sidebar.image(profile_pic, width=150, use_container_width=False, caption="Anmol Chaubey", output_format="PNG")
st.sidebar.markdown("""
    **Name:** Anmol Chaubey  
    **Email:** anmolchaubey820@gmail.com  
    [LinkedIn](https://www.linkedin.com/in/anmol-chaubey-120b42206/)
""")

assistant_name = "AskAI"
personality = st.sidebar.radio("Choose Assistant Personality", ["Formal", "Casual", "Humorous"], index=1)

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.write("Chat cleared.")

# Embedding Model and LLM
try:
    llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", max_tokens=1024, api_key=nvidia_api_key)
    document_embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", model_type="passage", api_key=nvidia_api_key)
except Exception as e:
    st.error(f"Failed to initialize NVIDIA services: {e}")
    st.stop()

# Vector Database Store
vector_store_path = "vectorstore.pkl"
raw_documents = DirectoryLoader(DOCS_DIR, glob="*.txt").load()  # Load only .txt files

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
                chunk_size=512,
                chunk_overlap=100,
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

# Updated prompt template for context-aware responses
prompt_template = ChatPromptTemplate.from_messages([
    ("system", f"You are a helpful AI assistant named {assistant_name}. You communicate in a {personality.lower()} tone. "
               "Given the context below, extract and summarize all the main topics. If no context is provided, respond accordingly.\n"
               "Context: {{context}}"),
    ("user", "{{input}}")
])

# Input for user prompt
user_input = st.text_area("Enter your prompt here:", "", height=100)

if st.button("Send") and user_input.strip():
    # Append user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Retrieve relevant documents if vectorstore exists
    if vectorstore:
        try:
            relevant_docs = vectorstore.similarity_search(user_input, k=3)
            context = "\n".join([f"- {doc.page_content.strip()}" for doc in relevant_docs if doc.page_content.strip()])
        except Exception as e:
            st.error(f"Error retrieving context: {e}")
            context = ""
    else:
        context = ""
    
    # Inform the user if no context is available
    if not context.strip():
        st.info("No relevant documents found in the knowledge base.")
    
    # Debugging: Show retrieved context
    st.write("Context provided:", context)
    
    # Format the prompt with context and input
    prompt = prompt_template.format_messages(context=context, input=user_input)
    
    # Invoke the LLM
    try:
        response = llm.invoke(prompt).content
    except Exception as e:
        st.error(f"Error generating response: {e}")
        response = "Sorry, I encountered an error while generating a response."
    
    # Post-process the response to remove unnecessary clutter
    def clean_response(response, max_length=500):
        response = response.strip()
        response = "\n".join([line.strip() for line in response.split("\n") if line.strip()])
        return response[:max_length] + "..." if len(response) > max_length else response

    cleaned_response = clean_response(response)
    
    # Append assistant's response to session state
    st.session_state.messages.append({"role": "assistant", "content": cleaned_response})
    with st.chat_message("assistant"):
        st.markdown(cleaned_response)
