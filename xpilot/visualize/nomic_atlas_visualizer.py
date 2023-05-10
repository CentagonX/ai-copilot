import os
import json
import numpy as np
from dotenv import load_dotenv
import nomic
from nomic import atlas

from xpilot.vector.pinecone import pinecone_service

def init_nomic():
    nomic.api_key = os.getenv("NOMIC_API_TOKEN")
    nomic.login(nomic.api_key)

def atlas_visualize_embeddings(index, id_list, name_space):
  
  ids = []
  embeddings = []
  vectors = pinecone_service.fetch_embeddings(index, id_list, name_space)
  for id, vector in vectors['vectors'].items():
      ids.append(id)
      embeddings.append(vector['values'])

  embeddings = np.array(embeddings)

  return atlas.map_embeddings(embeddings=embeddings, data=[{'id': id} for id in ids], id_field='id')

if __name__ == '__main__':
    
    load_dotenv()
    pinecone_service.init_pinecone()
    init_nomic()
    index = pinecone_service.get_index(pinecone_service.list_indexes()[0])
    print (index)
    ids=[str(i) for i in range(100)]
    print (ids)
    atlas_visualize_embeddings(index, ids, "chatgpt-demo")