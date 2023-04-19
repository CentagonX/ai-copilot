import os
import openai
from dotenv import load_dotenv 

load_dotenv()

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

print(openai.organization)
print(openai.api_key)

openai.models = openai.Model.list()
print(openai.models)