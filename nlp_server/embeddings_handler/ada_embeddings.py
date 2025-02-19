import openai
import os
from dotenv import load_dotenv
import settings as s
import numpy as np
from sentence_transformers import SentenceTransformer, util
from numpy import mean

from sklearn.metrics.pairwise import cosine_similarity


class AdaEmbeddings:
   dotenv_path = s.PROJECT_ROOT / '.env'
   load_dotenv(dotenv_path)
   open_ai_key = os.getenv('OPENAI_API_KEY')
   openai.api_key = open_ai_key
   client = openai.OpenAI()

   @staticmethod
   def get_embedding(text, model="text-embedding-3-small"):
      final_instance = []
      if not isinstance(text, list):
         text = [text]
      embedding = AdaEmbeddings.client.embeddings.create(input=text, model=model)
      for x in embedding.data:
         final_instance.append(np.array(x.embedding))
      return np.array(final_instance)


if __name__ == '__main__':
   Embedder = AdaEmbeddings()

   text = ['Hello there.' , 'General Kenobi.']
   query = ['Obi Wan.']
   embeddings = Embedder.get_embedding(text)
   query_embedding = Embedder.get_embedding(query)
   query_embedding2 = Embedder.get_embedding(' ')

   doc1_similarity = [cosine_similarity(query_embedding , vector.reshape(1 , -1)) for vector in embeddings]
   doc1_similarity_alter = util.cos_sim(query_embedding , embeddings)

   doc1_similarity_faster = np.dot(embeddings,query_embedding.T)

   doc1_similarity = mean([cosine_similarity(query_embedding , vector.reshape(1 , -1)) for vector in embeddings])



   mewo = 1