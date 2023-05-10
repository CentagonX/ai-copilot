import os
import openai
import io
import json
import requests
import textract
import pinecone
import numpy as np
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
from PyPDF2 import PdfReader, PdfWriter
from collections import defaultdict

def pdf_to_json(key, public_url):
    print("\nStarting PDF to JSON----")
    # Download the file from the public URL
    response = requests.get(public_url)
    # print('response docs', response)
    # print('response content', response.content)

    with io.BytesIO(response.content) as file_stream:
        file_stream.seek(0)

        case_pdf = PdfReader(file_stream)
        number_of_pages = len(case_pdf.pages)
        # print ('case_pdf', case_pdf)
        print ('number_of_pages', number_of_pages)

        output = []
        for i in range(number_of_pages):
            page = case_pdf.pages[i]
            # print('page', page)
            # print('text from page', page.extract_text())
            output.append(page.extract_text())

        pages = [" ".join(str(e)) for e in output]
        docs = dict({"document_id": key, "pages": pages})
        # print('docs', docs)

        print("-----Ending PDF to JSON\n")
        return docs


def generate_embeddings(input_data, embedding_config_path):
    print("\nStarting to generate embeddings----")
    openai.organization = os.getenv("OPENAI_ORG_ID")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    with open(embedding_config_path, 'r') as embedding_config_file:
        embedding_config_data = json.load(embedding_config_file)

    df = pd.DataFrame.from_dict(input_data)

    index_list = df.index.values.tolist()
    page_numbers = map (lambda x: x + 1, index_list)
    df["page_id"] = pd.DataFrame(page_numbers)

    embedding_model = embedding_config_data["model"]
    df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model))

    print("-----Ending generating embeddings\n")
    return df[["document_id","page_id","pages","ada_embedding"]].to_json(orient='table', index=False)


def upload_to_pinecone(input_data, pinecone_config_path, pinecone_id_path):
    print("\nStarting to pinecone upload----")
    pinecone.environment = os.getenv("PINECONE_ENVIRONMENT")
    pinecone.api_key = os.getenv("PINECONE_API_KEY")

    pinecone.init(api_key=pinecone.api_key, environment=pinecone.environment)
    active_indexes = pinecone.list_indexes()
    index = pinecone.Index(active_indexes[0])

    with open(pinecone_config_path, 'r') as pinecone_config_file:
        pinecone_config_data = json.load(pinecone_config_file)

    pinecone_counter_path = os.path.join(root_path, pinecone_id_path)
    pinecone_counter_file = open(pinecone_counter_path, 'r+')
    pinecone_counter_id = int(pinecone_counter_file.read())

    def generate_vector(row):
        return(row["document_id"], row["page_id"], row["pages"],  row["ada_embedding"])


    # print('input data is the following', input_data)
    df = pd.DataFrame.from_dict(json.loads(input_data)["data"])
    vectors_series = df.apply(lambda x: generate_vector(x), axis=1)
    # print('vector series', vectors_series)

    vectors = []
    batch_size = pinecone_config_data["batch_size"]
    if batch_size > len(vectors_series):
        batch_size = len(vectors_series)

    print ("The batch size ==> " + str(batch_size))
    print ("Start with pinecone counter id ==> " + str(pinecone_counter_id))

    for i, row in vectors_series.items():
        # Append the current row to vectors regardless of the index
        vectors.append({'id': str(pinecone_counter_id), 'values': row[3], 'metadata': {'document_id': row[0], 'page_id': row[1], 'page_content': row[2]}})

        print(" Id => " + str(pinecone_counter_id) + " Id Mod Batch Size => " + str(pinecone_counter_id % batch_size))
        if pinecone_counter_id % batch_size == 0:
            # print(pinecone_counter_id, vectors)
            index.upsert(vectors=vectors, batch_size=batch_size, namespace='chatgpt-demo')
            vectors = []

        pinecone_counter_id = pinecone_counter_id + 1

    if (len(vectors)):
        index.upsert(vectors=vectors, batch_size=batch_size, namespace='chatgpt-demo')

    print ("End with pinecone counter id ==> " + str(pinecone_counter_id))
    pinecone_counter_file.seek(0)
    pinecone_counter_file.write(str(pinecone_counter_id))
    pinecone_counter_file.truncate()
    pinecone_counter_file.close()
    print("-----Ending pinecode upload\n")
    pinecone.deinit()

def search_case_docs_from_file(pinecone_config_path, prompts_catalog_path):
    openai.organization = os.getenv("OPENAI_ORG_ID")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    pinecone.environment = os.getenv("PINECONE_ENVIRONMENT")
    pinecone.api_key = os.getenv("PINECONE_API_KEY")

    pinecone.init(api_key=pinecone.api_key, environment=pinecone.environment)
    active_indexes = pinecone.list_indexes()
    index = pinecone.Index(active_indexes[0])

    with open(pinecone_config_path, 'r') as pinecone_config_file:
        pinecone_config_data = json.load(pinecone_config_file)

    namespace = pinecone_config_data["namespace"]
    top_k = pinecone_config_data["top_k"]

    with open(prompts_catalog_path, 'r') as prompts_catalog_file:
        prompts_catalog = json.load(prompts_catalog_file)

    def search_case_docs(prompt):
        prompt_embedding = get_embedding(
            prompt,
            engine="text-embedding-ada-002"
        )
        return index.query(
            vector=prompt_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )

    for query in prompts_catalog["prompts"]:
        print("Prompt: " + query)
        print("\nRelevant Docs:")

        results = search_case_docs(query)
        print(results)

    # pinecone.deinit()