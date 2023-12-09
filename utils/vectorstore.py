from pathlib import Path
import chromadb
from chromadb.config import Settings 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA

from utils import rag_config
from utils import clean_text

def get_vector_db(embeddings):
  db = None
  if rag_config['chroma']['srvr_mode'] == 'in_memory' :
    persistent_directory= rag_config['chroma']['persistent_dir']
    db = Chroma(collection_name=rag_config['chroma']['collection_name'], 
                embedding_function = embeddings, 
                persist_directory = persistent_directory)
  else:
    chroma_settings = Settings(
                              chroma_api_impl='rest',
                              allow_reset=True
                              )
    headers = {'X-Chroma-Token':rag_config['chroma']['chroma_key']}
    ip_addr = rag_config['chroma']['chroma_srvr_ip']
    port    = rag_config['chroma']['chroma_srvr_port']
    client = chromadb.HttpClient(host= ip_addr, port=port,
                                 headers=headers,
                                 settings=chroma_settings)
    db = Chroma(client=client,collection_name=rag_config['chroma']['collection_name'], 
              embedding_function = embeddings)
    
  return db
  
  

def load_all_documents_not_csv_from_folder(folder_path: str, glob_pattern: str = ""):
    """
    This function loads all documents from a specified folder path that are not in CSV format and
    returns them as a list.
    """
    loader = DirectoryLoader(folder_path, glob=glob_pattern or "**/*.txt", use_multithreading=True)
    loaded_documents = loader.load()
    return loaded_documents

def get_retriever(collection_name, chunk_size, output_folder):
    parent_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ", ", "."],
        chunk_size=chunk_size, #chunk_size in no.of characters
        chunk_overlap=200 #no. of characters overlapping
        )

    # Use smaller chunks for getting more relevant context
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        separators=["\n\n", "\n", " ", ", ", "."],
        chunk_overlap=20
      )

    # The storage layer for the parent documents
    fs = LocalFileStore(f"{output_folder}/vector_store")
    store = create_kv_docstore(fs)
    embeddings = OpenAIEmbeddings()  # type: ignore
    vectorstore = Chroma(collection_name=collection_name,
                         embedding_function = embeddings,
                         persist_directory=rag_config['chroma']['persistent_dir'])
    retriever = ParentDocumentRetriever(
      vectorstore=vectorstore,
      docstore=store,
      child_splitter=child_splitter,
      parent_splitter=parent_splitter,
      )
    return retriever

def rebuild_retriever(folder_name, chunk_size, output_folder):
    docs = load_all_documents_not_csv_from_folder(folder_name)
    for doc in docs:
      doc.page_content = clean_text(doc.page_content)
    retriever = get_retriever(rag_config['chroma']['collection_name'], chunk_size, output_folder)
    retriever.add_documents(docs, ids=None)
    return retriever

