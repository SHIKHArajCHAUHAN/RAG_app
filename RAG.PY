# import streamlit as st
# from dotenv import load_dotenv
# import os
# import fitz  # PyMuPDF
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# # from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_openai import AzureOpenAIEmbeddings
# from langchain.chat_models import AzureChatOpenAI
# from langchain.chains import RetrievalQA
# from langchain.memory import ConversationBufferMemory

# # === Load environment variables ===
# load_dotenv()

# api_key = os.getenv("API_KEY")
# api_base = os.getenv("API_BASE")  # e.g. https://your-resource-name.openai.azure.com
# api_version = os.getenv("API_VERSION")  # e.g. 2023-05-15
# chat_deployment = os.getenv("DEPLOYMENT_NAME")  # your chat model deployment name
# embed_deployment = "text-embedding-ada-002"  # your embedding model deployment name

# # === Streamlit UI Setup ===
# st.set_page_config(page_title="📚 Memory-Aware RAG Assistant", layout="wide")
# st.title("📚 Memory-Aware RAG Q&A Assistant")

# # === Validate environment variables ===
# if not all([api_key, api_base, api_version, chat_deployment, embed_deployment]):
#     st.error("❌ Missing one or more Azure OpenAI environment variables.")
#     st.stop()

# # === Functions ===
# @st.cache_data
# def extract_text_from_pdf(pdf_file):
#     doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
#     return "\n".join([page.get_text() for page in doc])

# def create_vector_store(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = text_splitter.split_text(text)

#     # Initialize the embedding model
#     embedding_model = AzureOpenAIEmbeddings(
#         azure_deployment="text-embedding-ada-002",
#         openai_api_version=os.getenv("API_VERSION"),
#         azure_endpoint="https://gen-cim-eas-dep-genai-train-openai.openai.azure.com/",
#         api_key=os.getenv("API_KEY")
#     )

#     # Specify persist_directory to store vector DB locally
#     vector_store = Chroma.from_texts(
#         chunks,
#         embedding=embedding_model,
#         collection_name="rag_store",
#         persist_directory=".chromadb"  # ensure local persistence
#     )
#     vector_store.persist()  # save to disk
#     return vector_store

# def get_qa_chain(vectorstore):
#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True,
#         input_key="query",  # Ensure the correct input key 'query'
#         k=3
#     )

#     retriever = vectorstore.as_retriever()

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=AzureChatOpenAI(
#             openai_api_key=api_key,
#             openai_api_base=api_base,
#             openai_api_version=api_version,
#             deployment_name=chat_deployment,
#             temperature=0
#         ),
#         chain_type="stuff",
#         retriever=retriever,
#         memory=memory,
#         return_source_documents=False
#     )

#     return qa_chain

# # === Upload PDF ===
# pdf_file = st.file_uploader("📄 Upload a PDF Document", type=["pdf"])

# if pdf_file:
#     raw_text = extract_text_from_pdf(pdf_file)
#     st.success("✅ PDF processed successfully.")

#     if "vectorstore" not in st.session_state:
#         st.session_state.vectorstore = create_vector_store(raw_text)
#         st.session_state.qa_chain = get_qa_chain(st.session_state.vectorstore)
#         st.session_state.chat_history = []

#     st.subheader("💬 Ask a Question")
#     user_question = st.text_input("Type your question here:")

#     if user_question:
#         # Call the QA chain with the correct input structure (query: user_question)
#         response = st.session_state.qa_chain({"query": user_question})  # Pass the query key

#         # Store interaction in chat history
#         st.session_state.chat_history.append({
#             "user": user_question,
#             "bot": response
#         })

#         st.write(response)

#     # Show chat history
#     if st.session_state.chat_history:
#         with st.expander("📜 Chat History"):
#             for i, chat in enumerate(st.session_state.chat_history, 1):
#                 st.markdown(f"**You {i}:** {chat['user']}")
#                 st.markdown(f"**Bot {i}:** {chat['bot']}")
# else:
#     st.info("👆 Please upload a PDF file to begin.")
import streamlit as st
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# === Load environment variables ===
load_dotenv()

api_key = os.getenv("API_KEY")
api_base = os.getenv("API_BASE")  # e.g. https://your-resource-name.openai.azure.com
api_version = os.getenv("API_VERSION")  # e.g. 2023-05-15
chat_deployment = os.getenv("DEPLOYMENT_NAME")  # your chat model deployment name
embed_deployment = "text-embedding-ada-002"  # your embedding model deployment name

# === Streamlit UI Setup ===
st.set_page_config(page_title="📚 Memory-Aware RAG Assistant", layout="wide")
st.title("📚 Memory-Aware RAG Q&A Assistant")

# === Validate environment variables ===
if not all([api_key, api_base, api_version, chat_deployment, embed_deployment]):
    st.error("❌ Missing one or more Azure OpenAI environment variables.")
    st.stop()

# === Functions ===
@st.cache_data
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def create_vector_store(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    # Initialize the embedding model
    embedding_model = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version=os.getenv("API_VERSION"),
        azure_endpoint="https://gen-cim-eas-dep-genai-train-openai.openai.azure.com/",
        api_key=os.getenv("API_KEY")
    )

    # Specify persist_directory to store vector DB locally
    vector_store = Chroma.from_texts(
        chunks,
        embedding=embedding_model,
        collection_name="rag_store",
        persist_directory=".chromadb"  # ensure local persistence
    )
    vector_store.persist()  # save to disk
    return vector_store

def get_qa_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="query",  # Ensure the correct input key 'query'
        k=3
    )

    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=AzureChatOpenAI(
            openai_api_key=api_key,
            openai_api_base=api_base,
            openai_api_version=api_version,
            deployment_name=chat_deployment,
            temperature=0
        ),
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=False
    )

    return qa_chain

# === Upload PDF ===
pdf_file = st.file_uploader("📄 Upload a PDF Document", type=["pdf"])

if pdf_file:
    raw_text = extract_text_from_pdf(pdf_file)
    st.success("✅ PDF processed successfully.")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = create_vector_store(raw_text)
        st.session_state.qa_chain = get_qa_chain(st.session_state.vectorstore)
        st.session_state.chat_history = []

    st.subheader("💬 Ask a Question")
    user_question = st.text_input("Type your question here:")

    if user_question:
        # Call the QA chain with the correct input structure (query: user_question)
        response = st.session_state.qa_chain({"query": user_question})  # Pass the query key

        # Extract the relevant parts from the response:
        # Get the answer from the response (it's typically under 'result' in the structured response)
        answer = response.get('result', None)  # Make sure to check if this key exists
        
        if answer:
            # Store interaction in chat history
            st.session_state.chat_history.append({
                "user": user_question,
                "bot": answer  # Store only the answer text
            })

            # Display the question and the answer in a clean format
            st.markdown(f"**You asked:** {user_question}")
            st.markdown(f"**Answer:** {answer}")
        else:
            st.error("❌ No answer found.")

    # Show chat history
    if st.session_state.chat_history:
        with st.expander("📜 Chat History"):
            for i, chat in enumerate(st.session_state.chat_history, 1):
                st.markdown(f"**You {i}:** {chat['user']}")
                st.markdown(f"**Bot {i}:** {chat['bot']}")
else:
    st.info("👆 Please upload a PDF file to begin.")
