import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from datasets import Dataset
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import ArxivLoader
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from utils.utils import clean_text, create_a_folder
from vector_store.vectorstore import get_retriever
from templates import get_question_template, get_answer_template
from tqdm import tqdm

import configparser
import config

# Config Directory
PACKAGE_ROOT = Path(config.__file__).resolve().parent
#print(PACKAGE_ROOT)
CONFIG_FILE_PATH = PACKAGE_ROOT / "rag_config.ini"
#print(CONFIG_FILE_PATH)

rag_config = configparser.ConfigParser()
rag_config.read(CONFIG_FILE_PATH)

docs_count = int(rag_config['validate']['input_docs_count'])
output_folder = rag_config['validate']['output_folder']
create_validation_qa_set = bool(rag_config['validate']['create_validation_file_set'])
llm_chat = rag_config['DEFAULT']['llm_chat']
primary_qa_llm = ChatOpenAI(model_name=llm_chat, temperature=0)

def load_documents_from_arxiv():
    test_docs = ArxivLoader(query="Retrieval Augmented Generation", load_max_docs=docs_count).load()
    for doc in test_docs:
        doc.page_content = clean_text(doc.page_content)
    return test_docs

def get_validation_retriever():
    chroma_validation_dir = create_a_folder(output_folder, rag_config['validate']['chroma_loc'])
    validation_retriever = get_retriever(rag_config['validate']['collection_name'],
                                         int(rag_config['DEFAULT']['chunk_size']),
                                         chroma_validation_dir)

    return validation_retriever
    
def get_validation_qa_chain(validation_retriever):
    
    qa_chain = RetrievalQA.from_chain_type(
        primary_qa_llm,
        retriever=validation_retriever,
        return_source_documents=True
        )
    return qa_chain

# Create the validation dataset
def create_validation_dataset(validation_retriever, docs):
        
    if create_validation_qa_set == False:
        return
    
    docs = load_documents_from_arxiv()
    validation_retriever.from_documents(docs)
    
    #Step1: Use llm and content of docs as context, get the relevant questions from llm
    
    question_schema = ResponseSchema(
                        name="question",
                        description="A question about the context."
                        )

    question_response_schemas = [
        question_schema,
    ] 
    question_output_parser = StructuredOutputParser.from_response_schemas(question_response_schemas)
    format_instructions = question_output_parser.get_format_instructions()
    prompt_template = ChatPromptTemplate.from_template(template=get_question_template)
    
    # qac_triples contains question-context-answer triples
    qac_triples = []
    for doc in tqdm(docs):
        messages = prompt_template.format_messages(
            context=doc,
            format_instructions=format_instructions
            )
        response = primary_qa_llm(messages)
        try:
            output_dict = question_output_parser.parse(response.content)
        except Exception as e:
            continue
        output_dict["context"] = doc
        qac_triples.append(output_dict)
        
    # Step2: Use Question-Context and get LLM response as ground-truth
    
    answer_schema = ResponseSchema(
                        name="answer",
                        description="an answer to the question"
                    )

    answer_response_schemas = [
        answer_schema,
    ]

    answer_output_parser = StructuredOutputParser.from_response_schemas(answer_response_schemas)
    format_instructions = answer_output_parser.get_format_instructions()
    prompt_template = ChatPromptTemplate.from_template(template=get_answer_template)
    
    for triple in tqdm(qac_triples):
        messages = prompt_template.format_messages(
            context=triple["context"],
            question=triple["question"],
            format_instructions=format_instructions
        )
        response = primary_qa_llm(messages)
        try:
            output_dict = answer_output_parser.parse(response.content)
        except Exception as e:
            continue
        triple["answer"] = output_dict["answer"]

    # Step3: Save the q-a-c triples into a csv file
    ground_truth_qac_set = pd.DataFrame(qac_triples)
    ground_truth_qac_set["context"] = ground_truth_qac_set["context"].map(lambda x: str(x.page_content))
    ground_truth_qac_set = ground_truth_qac_set.rename(columns={"answer" : "ground_truth"})
    eval_dataset = Dataset.from_pandas(ground_truth_qac_set)
    eval_dataset.to_csv("groundtruth_eval_dataset.csv")