import os
import openai
import json
import pandas as pd
import tiktoken 

from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv 

load_dotenv()

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

print (openai.organization)
print (openai.api_key)

root_path = os.getcwd()
embedding_config = os.path.join(root_path, "config/embedding_model_config.json")

with open(embedding_config, 'r') as embedding_config_file:
    embedding_config_data = json.load(embedding_config_file)

print (embedding_config_data)

risks_dataset_path = os.path.join(root_path, "data/risks/risks.csv")

print (risks_dataset_path)

df = pd.read_csv(risks_dataset_path, header=0)                 
print (len(df))
print (df.columns)

df = df.drop('Risksignature.remediation', axis = 1)

unstructured_log_desc = []
remove = ['undefined', '[]', '{}', 'nan']
listDict = df.to_dict(orient='records')
for entry in listDict:
    sentence = "The risk identified has "
    for key in entry.keys():
        if str(entry.get(key)).strip() not in remove: 
            # print("The {} is {}".format(key, entry.get(key)))
            sentence += ("{} {} ".format(key, entry.get(key)))
    unstructured_log_desc.append(sentence)

df["logdesc"] = unstructured_log_desc
print (df['logdesc'])

encoding = tiktoken.get_encoding(embedding_config_data["embedding_encoding"])
max_tokens = embedding_config_data["max_tokens"]
embedding_model = embedding_config_data["model"]

# df["n_tokens"] = df.logdesc.apply(lambda x: len(encoding.encode(x)))
# df = df[df.n_tokens <= max_tokens]

df['ada_embedding'] = df.logdesc.apply(lambda x: get_embedding(x, engine=embedding_model))
print (df.head(2))

df.to_csv('data/embeddings/risk_data_with_embeddings.csv', index=False)