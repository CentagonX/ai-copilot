import os
import openai
import io
import json
import requests
import textract
import pinecone
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from xpilot.vector.pinecone import pinecone_service
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

def query_docs_for_prompts(pinecone_config_path, prompts_catalog_path):

    load_dotenv()
    pinecone_env = pinecone_service.init_pinecone()
    index = pinecone_service.get_index(pinecone_service.list_indexes()[0])

    openai.organization = os.getenv("OPENAI_ORG_ID")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    with open(pinecone_config_path, 'r') as pinecone_config_file:
        pinecone_config_data = json.load(pinecone_config_file)

    with open(prompts_catalog_path, 'r') as prompts_catalog_file:
        prompts_catalog = json.load(prompts_catalog_file)

    def search_docs(prompt):
        prompt_embedding = get_embedding(
            prompt,
            engine="text-embedding-ada-002"
        )
        return pinecone_service.query_embeddings_from_index(index, embeddings, pinecone_config_data)

    for query in prompts_catalog["prompts"]:
        print("Prompt: " + query)
        print("\nRelevant Docs:")

        results = search_docs(query)
        print(results)

    pinecone_service.deinit()