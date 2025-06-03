from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain

app = Flask(__name__)
CORS(app) 
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
DOCUMENT_PATH = "document.txt" 

# Load Document
def load_document(file_path):
    loader = TextLoader(file_path)
    return loader.load()

# Split Document into Chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

# Create Embeddings
def create_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in FAISS Vector Database
def store_embeddings(docs, embeddings):
    vector_db = FAISS.from_documents(docs, embeddings)
    vector_db.save_local("faiss_index")
    return vector_db

# Load FAISS Database
def load_vector_db(embeddings):
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Get Retriever
def get_retriever(vector_db):
    return vector_db.as_retriever(search_kwargs={"k": 3})

# Set Up Gemini 2.0 Flash with Context-Aware Retrieval
def setup_gemini(api_key, retriever):
    system_prompt = """
    You are a professional AI medical assistant specialized in eye-related conditions such as glaucoma, diabetic retinopathy, macular degeneration, retina issues, and fundus imaging.

    Please follow these guidelines:

    1. If document context is available and relevant, use it to answer the question accurately.
    2. If the document does not cover the topic but the question is eye-related, respond confidently using your own medical knowledge without mentioning the document at all.
    3. Never say things like “I can help you with...” or “The document does not mention...”. Be direct, professional, and informative.
    4. If the question is about general medicine, acknowledge your specialization in eye health but provide useful information if possible.
    5. If the question is not related to medicine, respond with: “I'm sorry, but I specialize only in eye-related medical queries.”
    6. If the user says something like "tell me more", "elaborate", or "give more information", always continue with more detail about the most recently discussed eye topic. Do not ask for clarification.
    7. If the user input is a short phrase or single medical term (e.g., "retinal image", "macula", "fundus image", "OCT"), treat it as a request for a definition or explanation and respond accordingly using your medical expertise.
    8. Always assume terms like retina, optic, vision, eye, macula, glaucoma, diabetic retinopathy, or similar are eye-related and should be answered confidently.

    Use the following context if available:

    {context}

    User: {input}

    Response:
    """

    prompt = PromptTemplate.from_template(system_prompt)
    llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    return create_retrieval_chain(retriever, prompt | llm)

documents = load_document(DOCUMENT_PATH)
docs = split_documents(documents)
embeddings = create_embeddings()
vector_db = store_embeddings(docs, embeddings)
retriever = get_retriever(load_vector_db(embeddings))
chatbot = setup_gemini(API_KEY, retriever)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "Empty input"}), 400

    response = chatbot.invoke({"input": user_input})

    chatbot_reply = response.get("answer", "I'm sorry, I couldn't find a relevant answer.")
    return jsonify({"response": chatbot_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
