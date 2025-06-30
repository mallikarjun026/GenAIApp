import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
#from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

# Load API Key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 1. Load all PDFs from folder using DirectoryLoader
def load_documents(folder_path):
    loader = DirectoryLoader(
        path=folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        recursive=True,
        show_progress=True,
    )
    return loader.lazy_load()

# 2. Split documents into overlapping chunks with metadata
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# 3. Create embeddings and store in FAISS
def embed_documents(docs):
    #embeddings = OpenAIEmbeddings()
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=1536  # optional, but keeps control over vector size
    )
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore

# 4. Create LangChain QA system (retriever + LLM)
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
   # llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# === Main function ===
def main():
    # folder_path = "pdfs"
    # folder_path = "samplePDF"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, "pdfs")

    print("\n📄 Loading documents...")
    raw_docs = load_documents(folder_path)

    print("✂️ Splitting documents...")
    split_docs = split_documents(raw_docs)

    print("🔍 Embedding and creating vector store...")
    vectorstore = embed_documents(split_docs)

    print("💬 Setting up QA chain...")
    qa_chain = create_qa_chain(vectorstore)
    while True:
        query = input("Ask your question: else say exit to quit ")
        if query.strip().lower() == "exit":
            print("👋 Exiting....!")
            break
        print("🤖 Getting your answer...\n")
        result = qa_chain(query)

        print("📌 Answer:\n", result["result"])

        # print("\n📚 Source Chunks From:")
        # for doc in result["source_documents"]:
        #     print(f"- {doc.metadata.get('source')} (Page: {doc.metadata.get('page', 'N/A')})")

if __name__ == "__main__":
    main()
