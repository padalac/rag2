import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import configparser
import config, vector_store
import time


from utils.utils import (
    create_a_folder,
    process_text,
    process_images,
    process_image_and_text_from_docx,
    get_all_image_descriptions
)

from vector_store.vectorstore import(
    rebuild_retriever,
    get_retriever
)



# Config Directory
PACKAGE_ROOT = Path(config.__file__).resolve().parent
#print(PACKAGE_ROOT)
CONFIG_FILE_PATH = PACKAGE_ROOT / "rag_config.ini"
#print(CONFIG_FILE_PATH)

rag_config = configparser.ConfigParser()
rag_config.read(CONFIG_FILE_PATH)

input_folder = rag_config['DEFAULT']['input_folder']
output_folder = rag_config['DEFAULT']['output_folder']
chunk_size = int(rag_config['DEFAULT']['chunk_size'])
llm_chat = rag_config['DEFAULT']['llm_chat']

def process_input_documents():
    # Create the Output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    text_files_path = ""
    image_files_path = ""
    # Traverse the input_folder and process each file
    t1_start = time.time()
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if file_name.endswith(".pdf"):
            with open(file_path, "rb") as f_pdf:
                text_files_path = process_text(file_path, output_folder, file_name)
                #image_files_path = process_images(file_path, output_folder)
        elif file_name.endswith(".docx"):
            text_files_path, image_files_path = process_image_and_text_from_docx(file_name, file_path, output_folder)
    t1_end = time.time()
    print("process_input_documents took time to complete -- ", t1_end - t1_start)
    return text_files_path, image_files_path

def get_qa_retriever(text_files_path, image_files_path):
    #get_all_image_descriptions(image_files_path, text_files_path)
    chroma_path = create_a_folder(output_folder, rag_config['chroma']['chroma_loc'])
    t2_start = time.time()
    qa_retriever = rebuild_retriever(text_files_path, chunk_size, chroma_path)
    t2_end = time.time()
    print("rebuild_retriever took time to complete -- ", t2_end - t2_start)
    return qa_retriever

if __name__ == "__main__":
    read_mode = True
    if rag_config['DEFAULT']['mode'] != 'read' :
        read_mode = False
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not set")
    
    chat_model = ChatOpenAI(model_name=llm_chat, temperature=0)
    chat_model.openai_api_key = openai_api_key
    #read_mode = True
    if read_mode == False:
        #upload documents and query them
        text_files_path, image_files_path = process_input_documents()
        qa_retriever = get_qa_retriever(text_files_path, image_files_path)

        rag_qa = RetrievalQAWithSourcesChain.from_chain_type(
                        llm=chat_model,
                        chain_type="stuff",
                        retriever=qa_retriever,
                        return_source_documents=True
                        )
        qa_retriever.vectorstore.persist()
        time.sleep(10)
        if rag_config['DEFAULT']['mode'] == 'update_only' :
            exit(0)
