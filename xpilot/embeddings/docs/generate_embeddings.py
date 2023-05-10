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
case_sample_dataset_path = os.path.join(root_path, "docs/sample/HortonTracy-12-14-2022-Original.json")

with open(case_sample_dataset_path, 'r') as case_sample_dataset_file:
    case_sample_dataset = json.load(case_sample_dataset_file)

df = pd.DataFrame.from_dict(case_sample_dataset)                 
print(df.head(4))

index_list = df.index.values.tolist()
# print (index_list)
page_numbers = map (lambda x: x + 1, index_list)
df["page_id"] = pd.DataFrame(page_numbers)

embedding_model = embedding_config_data["model"]
df['ada_embedding'] = df.pages.apply(lambda x: get_embedding(x, engine=embedding_model))
print (df.head(4))
df[["document_id","page_id","pages","ada_embedding"]].to_json('docs/embeddings/sample_pdf_embeddings.json', orient='table', index=False)