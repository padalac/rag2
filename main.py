import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import configparser
import config, vector_store
import streamlit as st
import time
import prometheus_client as prom
from prometheus_client import Counter

from vector_store.vectorstore import(
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

total_queries = prom.Counter('total_queries', 'Total no.of queries asked so far, by the users of this app')
non_empty_responses = prom.Counter('non_empty_responses', 'No. of times the answers were based on context')
empty_responses = prom.Counter('empty_responses', 'No.of times there was no context based response')

@st.cache_data
def get_val_results_from_file():
    validation_folder = rag_config['validate']['validation_file_loc']
    validation_folder_path = os.path.join(output_folder, validation_folder)
    validation_results_path = os.path.join(validation_folder_path, "evaluation_result.txt")

    val_results = ""
    with open(validation_results_path, "r") as fval:
        val_results = fval.read()
        print(val_results)

    data = {}
    for pair in val_results.lstrip('{').rstrip('}').split(','):
        key, value = pair.strip().split(':')
        data[key.strip('\'')] = float(value.strip())

    print(data)
    return data

@st.cache_data
def generate_query_response(_agent_chain, query):
    response = _agent_chain({"question": query}, return_only_outputs=False)
    return response

def main_qa():

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not set")
    
    chat_model = ChatOpenAI(model_name=llm_chat, temperature=0)
    chat_model.openai_api_key = openai_api_key

    chroma_path = os.path.join(output_folder, rag_config['chroma']['chroma_loc'])
    #use existing vectorDB to query results
    retriever = get_retriever(rag_config['chroma']['collection_name'], chunk_size, chroma_path)
    rag_qa = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=chat_model,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                    )

    st.title("Enterprise QnA chat bot")
    st.markdown("RAG with ChatGPT4 based Q and A application")
    query = ""
    query = st.text_input("Enter the query: ")
    print(query)
    if query != "":
        total_queries.inc()
        t3_start = time.time()
        response = generate_query_response(rag_qa, query)
        t3_end = time.time()
        qp_time_taken = t3_end - t3_start
        print("generate_query_response took time to complete -- ", qp_time_taken)
        print(response)
        st.markdown("\nResponse\n\n")
        st.markdown(response["answer"])
        st.markdown("\nSource Documents used:")

        if (response['sources']==''):
            st.markdown("There were no relevant source documents corresponding to this query")
            empty_responses.inc()
        else:
            if response['sources'].startswith("Output/Text"):
                file_list = response['sources'].split(",")
                file_names = []
                for file_name in file_list:
                    file_names.append(file_name.strip().lstrip(f"Output/Text/Text_").rstrip('.txt'))
                all_files = ", ".join(file_names)
                st.markdown(all_files)
                non_empty_responses.inc()
            else:
                st.markdown(response['sources'])
                empty_responses.inc()
        st.markdown("Time taken to complete query processing (secs):")
        st.markdown(qp_time_taken)
    

