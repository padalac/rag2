from pathlib import Path
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA

def load_all_documents_not_csv_from_folder(folder_path: str, glob_pattern: str = ""):
    """
    This function loads all documents from a specified folder path that are not in CSV format and
    returns them as a list.
    """
    loader = DirectoryLoader(folder_path, glob=glob_pattern or "**/*")
    loaded_documents = loader.load()
    return loaded_documents

def get_retriever(chunk_size, output_folder):
    parent_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ", ", "."],
        chunk_size=chunk_size, #chunk_size in no.of characters
        chunk_overlap=200 #no. of characters overlapping
        )

    # Use smaller chunks for getting more relevant context
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        separators=["\n\n", "\n", " ", ", ", "."],
        chunk_overlap=20
      )

    # The storage layer for the parent documents
    fs = LocalFileStore(f"{output_folder}/vector_store")
    store = create_kv_docstore(fs)
    embeddings = OpenAIEmbeddings()  # type: ignore
    vectorstore = Chroma(collection_name="all_documents", embedding_function = embeddings, persist_directory="../Chroma_DB/")
    retriever = ParentDocumentRetriever(
      vectorstore=vectorstore,
      docstore=store,
      child_splitter=child_splitter,
      parent_splitter=parent_splitter,
      )
    return retriever

def rebuild_retriever(folder_name, chunk_size, output_folder):
    docs = load_all_documents_not_csv_from_folder(folder_name)
    retriever = get_retriever(chunk_size, output_folder)
    retriever.add_documents(docs, ids=None)
    return retriever

