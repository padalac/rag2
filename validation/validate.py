import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
import pandas as pd
import datasets
from datasets import Dataset
from datasets import load_dataset
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import ArxivLoader
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from utils.utils import clean_text, create_a_folder
from vector_store.vectorstore import get_retriever
from validation.templates import get_question_template, get_answer_template
from tqdm import tqdm

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas.metrics.critique import harmfulness
from ragas import evaluate

import configparser
import config

# Config Directory
PACKAGE_ROOT = Path(config.__file__).resolve().parent
#print(PACKAGE_ROOT)
CONFIG_FILE_PATH = PACKAGE_ROOT / "rag_config.ini"
print(CONFIG_FILE_PATH)

rag_config = configparser.ConfigParser()
rag_config.read(CONFIG_FILE_PATH)

docs_count = int(rag_config['validate']['input_docs_count'])
output_folder = rag_config['validate']['output_folder']
create_validation_qa_set = bool(rag_config['validate']['create_validation_file_set'])
validation_folder = rag_config['validate']['validation_file_loc']
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
def create_validation_dataset(docs):
    
    '''
    validation_folder_path = create_a_folder(output_folder, validation_folder)    
    validation_file = os.path.join(validation_folder_path, "groundtruth_eval_dataset.csv")
    
    if create_validation_qa_set == False:
        return validation_file
    
    docs = load_documents_from_arxiv()
    validation_retriever.from_documents(docs)
    '''
    
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

    #eval_dataset.to_csv(validation_file)
    return eval_dataset

def get_dataset_from_csv_file(csv_file):
    # print(csv_file)
    dataset = datasets.load_dataset('csv', data_files=f'{csv_file}')
    # with open('eval_ds.txt', 'w') as f:
    #     print(dataset, file=f)
    return dataset

    
# Evaluating RAG pipelines (using RAGAS)
def create_ragas_dataset(rag_pipeline, eval_dataset):
    rag_dataset = []
    for row in tqdm(eval_dataset):
        answer = rag_pipeline({"query" : row["question"]})
        rag_dataset.append(
            {"question" : row["question"],
            "answer" : answer["result"],
            "contexts" : [context.page_content for context in answer["source_documents"]],
            "ground_truths" : [row["ground_truth"]]
            }
        )
    rag_df = pd.DataFrame(rag_dataset)
    rag_eval_dataset = Dataset.from_pandas(rag_df)
    return rag_eval_dataset

def evaluate_ragas_dataset(ragas_dataset):
    result = evaluate(
        ragas_dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
    )
    return result

def get_validation_result():
    
    validation_folder_path = create_a_folder(output_folder, validation_folder)    
    validation_file = os.path.join(validation_folder_path, "groundtruth_eval_dataset.csv")
    validation_retriever = get_validation_retriever()
    eval_dataset = ""
    
    #create_validation_qa_set = bool(False)

    #print(create_validation_qa_set)

    if create_validation_qa_set == True:
        docs = load_documents_from_arxiv()
        validation_retriever.add_documents(docs)
        eval_dataset = create_validation_dataset(docs)
        eval_dataset.to_csv(validation_file)
    else:
        eval_dataset = get_dataset_from_csv_file(validation_file)

    validation_qa_chain = get_validation_qa_chain(validation_retriever)
    qa_ragas_dataset = create_ragas_dataset(validation_qa_chain, eval_dataset)

#    with open('qa_ragas_ds.txt', 'w') as f:
#        print(qa_ragas_dataset, file=f)

    validation_result_file = os.path.join(validation_folder_path, "evaluation_result.csv")
    qa_ragas_dataset.to_csv(validation_result_file)
    evaluation_result = evaluate_ragas_dataset(qa_ragas_dataset)
    
    print(evaluation_result)
    
    return evaluation_result
    
if __name__ == "__main__":

    result = get_validation_result()
    print(result)
