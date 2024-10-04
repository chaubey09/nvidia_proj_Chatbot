# AskAI - A Custom Knowledge-Based Chatbot

This project is a Streamlit-powered chatbot application that allows users to chat with an AI assistant named **AskAI**. The chatbot is capable of answering user queries and can enhance its knowledge base by processing and storing documents. It also offers the ability to change the assistant's personality between **Formal**, **Casual**, and **Humorous**.

## Features

- **Document Upload**: Add new files to enhance the knowledge base of the chatbot.
- **Chatbot Personalities**: Choose between different assistant personalities: Formal, Casual, and Humorous.
- **Document-based Contextual Responses**: The assistant can retrieve context from uploaded documents to answer questions.
- **Dynamic Knowledge Store**: Utilizes FAISS for vector storage and retrieval, allowing efficient document embedding and query answering.
- **Profile Section**: Displays user contact information with a circular framed profile picture.

## Setup and Installation

### Requirements

- Python 3.x
- [Streamlit](https://streamlit.io/)
- [LangChain](https://github.com/hwchase17/langchain) 
- [NVIDIA AI Endpoints](https://developer.nvidia.com/nvidia-ai) (API key required)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Pillow](https://python-pillow.org/) (For image handling)
- [dotenv](https://pypi.org/project/python-dotenv/) (For environment variable management)

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/askai.git
    cd askai
    ```

2. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Add your NVIDIA API Key**:
   - Create a `.env` file in the project root.
   - Add the following line, replacing `your_api_key` with your actual API key:
     ```bash
     NVIDIA_API_KEY=your_api_key
     ```

4. **Place your profile picture**:
    - Save your profile picture as `profile_photo.png` in the project root directory.

### Run the App

Run the Streamlit app using the following command:

```bash
streamlit run app.py
