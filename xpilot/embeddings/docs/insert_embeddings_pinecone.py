import os
import json
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

root_path = os.getcwd()
pinecone_config_path = os.path.join(root_path, "config/pinecone_config.json")

with open(pinecone_config_path, 'r') as pinecone_config_file:
    pinecone_config_data = json.load(pinecone_config_file)

print(pinecone_config_data)
print(pinecone_config_data["batch_size"])

pinecone_counter_path = os.path.join(root_path, "config/pinecone_counter.id")
pinecone_counter_file = open(pinecone_counter_path, 'r+')
pinecone_counter_id = int(pinecone_counter_file.read())

sample_case_embeddings_path = os.path.join(root_path, "docs/embeddings/sample_pdf_embeddings.json")
# sample_case_embeddings_path = os.path.join(root_path, "docs/embeddings/sample_invoice_embeddings.json")

with open(sample_case_embeddings_path, 'r') as sample_case_embeddings_file:
    sample_case_embeddings = json.load(sample_case_embeddings_file)

def generate_vector(row):

    return(row["document_id"], row["page_id"], row["pages"],  row["ada_embedding"])

df = pd.DataFrame(sample_case_embeddings["data"])                 
print(df.head(4))

vectors_series = df.apply(lambda x: generate_vector(x), axis = 1)
print (vectors_series.head(2))

vectors = [] 
j = 1
batch_size = pinecone_config_data["batch_size"]
if batch_size > len(vectors_series):
    batch_size = len(vectors_series)

print ("The batch size ==> " + str(batch_size))
print ("Pinecone counter id ==> " + str(pinecone_counter_id))

for i, row in vectors_series.items():
    # Append the current row to vectors regardless of the index
    vectors.append({'id': str(pinecone_counter_id), 'values': row[3], 'metadata': {'document_id': row[0], 'page_id': row[1], 'page_content': row[2]}})
    
    print(" Id => " + str(pinecone_counter_id) + " Id Mod Batch Size => " + str(pinecone_counter_id % batch_size))
    if pinecone_counter_id % batch_size == 0:
        print(pinecone_counter_id, vectors)
        index.upsert(vectors=vectors, batch_size=batch_size, namespace='chatgpt-demo')
        vectors = []
    
    pinecone_counter_id = pinecone_counter_id + 1
    

if (len(vectors)):
    index.upsert(vectors=vectors, batch_size=batch_size, namespace='chatgpt-demo')


print ("Pinecone counter id to write ==> " + str(pinecone_counter_id))
pinecone_counter_file.seek(0)
pinecone_counter_file.write(str(pinecone_counter_id))
pinecone_counter_file.truncate()
pinecone_counter_file.close()