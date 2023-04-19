import os
import pinecone
import itertools
import pandas as pd
import numpy as np

from openai.embeddings_utils import get_embedding, cosine_similarity

from dotenv import load_dotenv 

load_dotenv()

pinecone.environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone.api_key = os.getenv("PINECONE_API_KEY")

pinecone.init(api_key=pinecone.api_key, environment=pinecone.environment)
index = pinecone.Index("co-pilot-demo")

datafile_path = "data/embedded/risk_data_with_embeddings.csv"

df = pd.read_csv(datafile_path)
df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)

product_embedding = []

# search through the reviews for a specific product
def search_risks(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002"
    )
    df["similarity"] = df.ada_embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("RisksignatureName: ", "")
        .str.replace("; RisksignatureDescription:", ": ")
        .str.replace("likelihoodLabel: ", "Likelihood: ")
        .str.replace("impactLabel: ", "Impact: ")
    )
    if pprint:
        for r in results:
            print(r[:200])
            print()
    return results

prompt = "What are the top high impact risks in my environment? Do not explain ..Just list them as a table." 

print("Prompt: " + prompt)
print("")
print("Search Results: ")
results = search_risks(df, prompt, n=10)