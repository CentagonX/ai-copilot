import os
import json
import openai
import pinecone
import pandas as pd
import numpy as np

from openai.embeddings_utils import get_embedding, cosine_similarity

from dotenv import load_dotenv 

load_dotenv()

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

print (openai.organization)
print (openai.api_key)

pinecone.environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone.api_key = os.getenv("PINECONE_API_KEY")

pinecone.init(api_key=pinecone.api_key, environment=pinecone.environment)
active_indexes = pinecone.list_indexes()

index = pinecone.Index(active_indexes[0])
print (pinecone.describe_index(active_indexes[0]))

root_path = os.getcwd()
pinecone_config = os.path.join(root_path, "config/pinecone_config.json")

with open(pinecone_config, 'r') as pinecone_config_file:
    pinecone_config_data = json.load(pinecone_config_file)

print(pinecone_config_data)
namespace = pinecone_config_data["namespace"]
# prompt = pinecone_config_data["query_prompt"]
top_k = pinecone_config_data["top_k"]

prompts_catalog_config = os.path.join(root_path, "config/prompts_catalog.json")

with open(prompts_catalog_config, 'r') as prompts_catalog_file:
    prompts_catalog = json.load(prompts_catalog_file)

def search_case_docs(prompt, pprint=True):
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
    print("")
    print("Relevant Docs: ")

    results = search_case_docs(query)
    print(results)