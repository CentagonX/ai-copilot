import os
import pinecone
import pandas as pd
import numpy as np

from dotenv import load_dotenv 

load_dotenv()

pinecone.environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone.api_key = os.getenv("PINECONE_API_KEY")

print (pinecone.environment)
print (pinecone.api_key)

pinecone.init(api_key=pinecone.api_key, environment=pinecone.environment)

active_indexes = pinecone.list_indexes()
print (active_indexes)

index = pinecone.Index(active_indexes[0])

def generate_vector(row):

    return(row["id"], row["signatureid"], row["riskScore"], row["likelihoodLabel"], row["impactLabel"], row["Risksignature.description"], row["ada_embedding"])

datafile_path = "data/embeddings/risk_data_with_embeddings.csv"

df = pd.read_csv(datafile_path)
df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)
print (df.head(2))

vectors_series = df.apply(lambda x: generate_vector(x), axis = 1)
print (vectors_series.head(2))
print (len(vectors_series))


vectors = [] 
j = 1
batch_size = 100
for i, row in vectors_series.items():
    if i == 0: continue
    if i % batch_size != 0:
        vectors.append({'id': str(row[0]), 'values': row[6], 'metadata': {'signatureid': row[1], 'riskScore': row[2], 'likelihoodLabel': row[3], 'impactLabel': row[4], 'riskSignatureDescription': row[5]}})
    else:
        print(i, vectors)
        index.upsert(vectors=vectors, batch_size=batch_size, namespace='security-co-pilot-demo')
        vectors = []
