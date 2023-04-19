import os
import openai
import json

from dotenv import load_dotenv 

load_dotenv()

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")

root_path = os.getcwd()
model_config = os.path.join(root_path, "config/gpt_model_config.json")
prompts_config = os.path.join(root_path, "config/prompts_config.json")

with open(model_config, 'r') as model_config_file:
    model_config_data = json.load(model_config_file)

print (model_config_data)
print (model_config_data["models"][0])
with open(prompts_config, 'r') as prompts_config_file:
    prompts_config_data = json.load(prompts_config_file)

print (prompts_config_data)
print (prompts_config_data['prompts'][0]['roles'])

prompts = [prompts_config_data['prompts'][0]['roles'][0]]
prompts.append(prompts_config_data['prompts'][0]['roles'][1])

response = openai.ChatCompletion.create(
    model = model_config_data["models"][0]["model"],
    max_tokens = model_config_data["models"][0]["max_tokens"],
    temperature = model_config_data["models"][0]["temperature"],
    messages = prompts
)
print (response)