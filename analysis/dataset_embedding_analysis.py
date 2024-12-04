import pickle

path = '/Users/max/projects/axiomatic-reranking/_axioms_cache_embeddings/embeddings_data_store.pkl'
with open(path, 'rb') as f:
    argu_embedding_dict = pickle.load(f)

