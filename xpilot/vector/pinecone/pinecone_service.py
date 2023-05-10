import os
import json
import pinecone

def init_pinecone():
  
  pinecone.environment = os.getenv("PINECONE_ENVIRONMENT")
  pinecone.api_key = os.getenv("PINECONE_API_KEY")
  pinecone.init(api_key=pinecone.api_key, environment=pinecone.environment)

  print (pinecone.environment)
  print (pinecone.api_key)

def list_indexes():

  return pinecone.list_indexes()


def get_index(idx):
  
  return pinecone.Index(idx)


def delete_embeddings(index, id_list, name_space):

  try:
    response = index.delete(
      ids=id_list,
      namespace=name_space
    )
    return response.status_code
  except Exception as ex:
    print (ex)


def fetch_embeddings(index, id_list, name_space):

  try:
    response = index.fetch(
        ids=id_list,
        namespace=name_space
    )
    return response
  except Exception as ex:
    print (ex)