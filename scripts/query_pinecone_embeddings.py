import os
import pinecone
import pandas as pd
import numpy as np

from openai.embeddings_utils import get_embedding, cosine_similarity

from dotenv import load_dotenv 

load_dotenv()

pinecone.environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone.api_key = os.getenv("PINECONE_API_KEY")

pinecone.init(api_key=pinecone.api_key, environment=pinecone.environment)
active_indexes = pinecone.list_indexes()
print (active_indexes)

index = pinecone.Index(active_indexes[0])


prompt = "What are the top high impact risks in my environment?" 

print("Prompt: " + prompt)
print("")
print("Search Results: ")

def search_risks(prompt, pprint=True):
    prompt_embedding = get_embedding(
        prompt,
        engine="text-embedding-ada-002"
    )
    return index.query(
        vector=prompt_embedding,
        top_k=3,
        include_metadata=True,
        namespace="security-co-pilot-demo"
    )

results = search_risks(prompt)
print(results)