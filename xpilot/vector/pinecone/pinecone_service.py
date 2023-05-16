import os
import json
import pinecone

class Environ():
  def __init__(self, data):
    self._data = data

  def getdata():

    return self._data

  def getenv():

    return self._data["PINECONE_ENVIRONMENT"]
  
  def getapikey():

    return self._data["PINECONE_API_KEY"]

def init():
  
  try:
    pinecone.environment = os.getenv("PINECONE_ENVIRONMENT")
    pinecone.api_key = os.getenv("PINECONE_API_KEY")
    pinecone.init(api_key=pinecone.api_key, environment=pinecone.environment)

    print (pinecone.environment)
    print (pinecone.api_key)
    data = {}
    data["PINECONE_ENVIRONMENT"] = pinecone.environment
    data["PINECONE_API_KEY"] = pinecone.api_key
    return Environ(data)
  except Exception as ex:
    print (ex)

def deinit():

  try:
    pinecone.deinit()
  except Exception as ex:
    print (ex)

def list_indexes():

  return pinecone.list_indexes()


def get_index(idx):
  
  return pinecone.Index(idx)


def delete_embeddings_from_index(index, id_list, name_space):

  try:
    response = index.delete(
      ids=id_list,
      namespace=name_space
    )
    return response.status_code
  except Exception as ex:
    print (ex)


def fetch_embeddings_from_index(index, id_list, name_space):

  try:
    response = index.fetch(
        ids=id_list,
        namespace=name_space
    )
    return response
  except Exception as ex:
    print (ex)

def query_embeddings_from_index(index, embeddings, pinecone_config_data):
  namespace = pinecone_config_data["namespace"]
  top_k = pinecone_config_data["top_k"]

  return index.query(
      vector=embedding,
      top_k=top_k,
      include_metadata=True,
      namespace=namespace
  )

def upload_embeddings_to_index(env, index, embeddings, pinecone_config_path, pinecone_id_path):

  with open(pinecone_config_path, 'r') as pinecone_config_file:
      pinecone_config_data = json.load(pinecone_config_file)

  pinecone_counter_path = os.path.join(root_path, pinecone_id_path)
  pinecone_counter_file = open(pinecone_counter_path, 'r+')
  pinecone_counter_id = int(pinecone_counter_file.read())

  def generate_vector(row):
      return(row["document_id"], row["page_id"], row["pages"],  row["ada_embedding"])

  # print('input data is the following', input_data)
  df = pd.DataFrame.from_dict(json.loads(embeddings)["data"])
  vectors_series = df.apply(lambda x: generate_vector(x), axis=1)
  # print('vector series', vectors_series)

  vectors = []
  batch_size = pinecone_config_data["batch_size"]
  if batch_size > len(vectors_series):
      batch_size = len(vectors_series)

  print ("The batch size ==> " + str(batch_size))
  print ("Start with pinecone counter id ==> " + str(pinecone_counter_id))

  for i, row in vectors_series.items():
      # Append the current row to vectors regardless of the index
      vectors.append({'id': str(pinecone_counter_id), 'values': row[3], 'metadata': {'document_id': row[0], 'page_id': row[1], 'page_content': row[2]}})

      print(" Id => " + str(pinecone_counter_id) + " Id Mod Batch Size => " + str(pinecone_counter_id % batch_size))
      if pinecone_counter_id % batch_size == 0:
          # print(pinecone_counter_id, vectors)
          index.upsert(vectors=vectors, batch_size=batch_size, namespace='chatgpt-demo')
          vectors = []

      pinecone_counter_id = pinecone_counter_id + 1

  if (len(vectors)):
      index.upsert(vectors=vectors, batch_size=batch_size, namespace='chatgpt-demo')

  print ("End with pinecone counter id ==> " + str(pinecone_counter_id))
  pinecone_counter_file.seek(0)
  pinecone_counter_file.write(str(pinecone_counter_id))
  pinecone_counter_file.truncate()
  pinecone_counter_file.close()
  print("-----Ending pinecode upload\n")
  pinecone.deinit()