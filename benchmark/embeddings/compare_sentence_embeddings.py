import os
import csv
import json
import torch
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv 
from tabulate import tabulate

load_dotenv()

root_path = os.getcwd()
prompts_catalog_config = os.path.join(root_path, "config/prompts_catalog.json")

with open(prompts_catalog_config, 'r') as prompts_catalog_file:
    prompts_catalog = json.load(prompts_catalog_file)

risks_log_corpus_path = os.path.join(root_path, "data/risks/risk_log_openai_text_corpus_1.csv")

corpus = []
with open(risks_log_corpus_path, 'r') as risks_log_corpus_file:
    corpus_reader = csv.reader(risks_log_corpus_file, delimiter=",")
    for reader in corpus_reader:
        for line in reader:
            corpus.append(line)

transformers_config_path = os.path.join(root_path, "config/transformers_comparisions.json")

with open(transformers_config_path, 'r') as transformers_config_file:
    transformers_config = json.load(transformers_config_file)

print (transformers_config["transformers"])
print (transformers_config["comparisions"])

queries = []
for query in prompts_catalog["prompts"]:
    queries.append(query)

print (queries)
print("")

queryComparisionsDF = pd.DataFrame(np.empty((0, 21)))
query_results_dict = {}
query_results_list = [{}]
# Find the closest top_k sentences of the corpus for each query sentence based on cosine similarity
top_k = min(1, len(corpus))
for query in queries:
    
    for transformer_model in transformers_config["transformers"]:

        model = SentenceTransformer(transformer_model)
        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
        query_embedding = model.encode(query, convert_to_tensor=True)

        for comparision in transformers_config["comparisions"]:
            
            key = transformer_model + "_" + comparision

            if comparision == 'cos_sim':
                value = []
                relevantCorpus = []
                # We use cosine-similarity and torch.topk to find the top_k scores
                cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                top_cos_results = torch.topk(cos_scores, k=top_k)
                for score, idx in zip(top_cos_results[0], top_cos_results[1]):
                    value.append(float("{:.4f}".format(score)))
                    relevantCorpus.append(corpus[idx])
                query_results_dict[key] = value
                query_results_dict["Relevant Corpus:"+ key] = relevantCorpus
            elif comparision == 'dot_product':
                value = []
                relevantCorpus = []
                dot_scores = util.dot_score(query_embedding, corpus_embeddings)[0]
                top_dot_results = torch.topk(dot_scores, k=top_k)
                for score, idx in zip(top_dot_results[0], top_dot_results[1]):
                    value.append(float("{:.4f}".format(score)))
                    relevantCorpus.append(corpus[idx])
                query_results_dict[key] = value
                query_results_dict["Relevant Corpus:"+ key] = relevantCorpus
        
    query_results_dict.update({"Query Name": query})
    print(query_results_dict)
    query_results_list.append(query_results_dict)

    if (queryComparisionsDF.empty):
        queryComparisionsDF.columns = list(query_results_dict.keys())

    # print(queryComparisionsDF)
    # print("Number of Columns: " + str(len(queryComparisionsDF.columns)))
    queryresultsDF = pd.DataFrame(query_results_dict)
    # print(queryresultsDF)
    # print("Number of Columns: " + str(len(queryresultsDF.columns)))

    queryComparisionsDF = pd.concat([queryComparisionsDF, queryresultsDF], axis=0) 

print(queryComparisionsDF)

queryComparisionsDF.to_csv('data/tests/sentence_transformer_comparision_results.csv', index=False)