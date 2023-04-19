import os
import openai
import json
import pandas as pd
import tiktoken 
import numpy as np

from openai.embeddings_utils import get_embedding
from dotenv import load_dotenv 

load_dotenv()

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

print (openai.organization)
print (openai.api_key)

root_path = os.getcwd()
risks_dataset_path = os.path.join(root_path, "data/risks/risks.csv")
sanitized_risks_dataset_path = os.path.join(root_path, "data/risks/sanitized_risks.csv")

print (risks_dataset_path)
sanitized = []
def sanitize_and_write_log_line(df):
    stringified = df.to_string(header=True,
                  index=False,
                  index_names=False).split('\n')
    
    for ele in stringified:
        sanitized.append(ele.replace("undefined", "").replace("NaN", ""))

    print(sanitized[0])
    ndf = pd.DataFrame(sanitized) 
    return ndf
        


df = pd.read_csv(risks_dataset_path, header=0)
df = df.drop('Risksignature.remediation', axis = 1)
print (df.head(2))

ndf = sanitize_and_write_log_line(df)
print (ndf.head(2))
ndf.to_csv(sanitized_risks_dataset_path, encoding='utf-8', index=False)
