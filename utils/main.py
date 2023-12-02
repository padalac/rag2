import os
import shutil
import getpass
from openai import OpenAI
import base64
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA

from utils import (
    create_a_folder,
    process_text,
    process_images,
    process_image_and_text_from_docx,
    get_all_image_descriptions
)

from agents import (
    get_tools,
    get_prompt_template,
    get_agent_chain_with_memory
)

from vectorstore import(
    rebuild_retriever
)

input_folder = "../input_docs"
print(input_folder)
output_folder = "../Output"
chunk_size = 7500

def process_input_documents():
    # Create the Output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(output_folder)

    text_files_path = ""
    image_files_path = ""
    # Traverse the input_folder and process each file
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if file_name.endswith(".pdf"):
            with open(file_path, "rb") as f_pdf:
                print(file_path)
                #text_files_path = process_text(file_path, output_folder, os.path.splitext(file_name)[0])
                text_files_path = process_text(file_path, output_folder, file_name)
                image_files_path = process_images(file_path, output_folder)
        elif file_name.endswith(".docx"):
            text_files_path, image_files_path = process_image_and_text_from_docx(file_name, file_path, output_folder)
            print(text_files_path)
        
    return text_files_path, image_files_path

def get_qa_retriever(text_files_path, image_files_path):
    get_all_image_descriptions(image_files_path, text_files_path)
    chroma_loc = create_a_folder(output_folder, "Chroma")
    #qa_retriever = get_retriever(text_files_path)
    qa_retriever = rebuild_retriever(text_files_path, chunk_size, chroma_loc)
    return qa_retriever

def generate_query_response(agent_chain, query, max_length=2000):
    #response = agent_chain.run(input=query)
    response = agent_chain({"input": query})
    return response

if __name__ == "__main__":

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is not set")
    
    serp_api_key = os.getenv("SERPAPI_API_KEY")
    if serp_api_key is None:
        raise ValueError("SERPAPI_API_KEY is not set")
    #os.environ["SERPAPI_API_KEY"] = getpass.getpass('Enter Google SERP API Key:')
    
    chat_model = ChatOpenAI(model_name="gpt-4-1106-preview")
    chat_model.openai_api_key = openai_api_key
    
    text_files_path, image_files_path = process_input_documents()
    qa_retriever = get_qa_retriever(text_files_path, image_files_path)
    
    rag_qa = RetrievalQA.from_chain_type(
                    llm=chat_model,
                    chain_type="stuff",
                    retriever=qa_retriever,
                    return_source_documents=True,
                    )
    tools = get_tools(rag_qa)
    prompt_template = get_prompt_template(tools)
    agent_chain = get_agent_chain_with_memory(chat_model, prompt_template, tools)
    
    # Clean up by deleting the folders and files created
    #shutil.rmtree(output_folder, ignore_errors=True)
    
    query = "what makes RAG superior?"
    print(generate_query_response(agent_chain, query))
    query = "Any other ways to improve its efficiency?"
    print(generate_query_response(agent_chain, query))
