import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
import pickle
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image

# Fetch the NVIDIA API key from st.secrets
nvidia_api_key = st.secrets["nvidia_api_key"]

if not nvidia_api_key:
    raise ValueError("NVIDIA API Key not found! Make sure it's set in the secrets.toml file.")

st.set_page_config(layout="wide")

# Sidebar for document upload and contact info
with st.sidebar:
    st.subheader("Add to the Knowledge Base")

    DOCS_DIR = os.path.abspath("./uploaded_docs")
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    # Upload files form
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:", accept_multiple_files=True)
        submitted = st.form_submit_button("Upload!")

    if uploaded_files and submitted:
        for uploaded_file in uploaded_files:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            with open(os.path.join(DOCS_DIR, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.read())

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
st.sidebar.image(profile_pic, width=150, use_column_width=False, caption="Anmol Chaubey", output_format="PNG")

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
llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", max_tokens=1024, api_key=nvidia_api_key)
document_embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", model_type="passage", api_key=nvidia_api_key)

# Vector Database Store
with st.sidebar:
    use_existing_vector_store = st.radio("Use existing vector store if available", ["Yes", "No"], horizontal=True)

vector_store_path = "vectorstore.pkl"
raw_documents = DirectoryLoader(DOCS_DIR).load()

vector_store_exists = os.path.exists(vector_store_path)
vectorstore = None
if use_existing_vector_store == "Yes" and vector_store_exists:
    with open(vector_store_path, "rb") as f:
        vectorstore = pickle.load(f)
    with st.sidebar:
        st.success("Existing vector store loaded successfully.")
else:
    with st.sidebar:
        if raw_documents and use_existing_vector_store == "Yes":
            with st.spinner("Splitting documents into chunks..."):
                text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=200)
                documents = text_splitter.split_documents(raw_documents)

            with st.spinner("Adding document chunks to vector database..."):
                vectorstore = FAISS.from_documents(documents, document_embedder)

            with st.spinner("Saving vector store"):
                with open(vector_store_path, "wb") as f:
                    pickle.dump(vectorstore, f)
            st.success("Vector store created and saved.")
        else:
            st.warning("No documents available to process!", icon="⚠️")

# Chat Interface
st.subheader(f"Chat with {assistant_name} ({personality} Mode)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt_template = ChatPromptTemplate.from_messages([
    ("system", f"You are a helpful AI assistant named {assistant_name}. You communicate in a {personality.lower()} tone. If provided with context, use it to inform your responses. If no context is available, use your general knowledge to provide a helpful response."),
    ("human", "{input}")
])

chain = prompt_template | llm | StrOutputParser()

# User input field
user_input = st.chat_input("Ask something...")

# Handle user input and generate response
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if vectorstore is not None and use_existing_vector_store == "Yes":
            retriever = vectorstore.as_retriever()
            docs = retriever.invoke(user_input)
            context = "\n\n".join([doc.page_content for doc in docs])
            augmented_user_input = f"Context: {context}\n\nQuestion: {user_input}\n"
        else:
            augmented_user_input = f"Question: {user_input}\n"

        # Show typing indicator while response is being generated
        with st.spinner(f"{assistant_name} is thinking..."):
            for response in chain.stream({"input": augmented_user_input}):
                full_response += response
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Add emoji reaction option for user to give feedback
    st.markdown("#### How was the response?")
    if st.button("👍"):
        st.success("Thanks for your feedback!")
    if st.button("👎"):
        st.error("Sorry to hear that, we'll strive to improve!")
