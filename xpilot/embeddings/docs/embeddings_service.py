import os
import openai
import json
import pandas as pd

from openai.embeddings_utils import get_embedding

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
