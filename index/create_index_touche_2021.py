import pyterrier as pt
import settings as s
import shutil
import os
from loguru import logger

if not pt.java.started():
    pt.java.init()

from pyterrier.datasets import get_dataset

x = pt.datasets.list_datasets()

def delete_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Directory '{path}' and all its contents have been deleted.")
    else:
        print(f"Directory '{path}' does not exist.")


if __name__ == '__main__':
    s.set_data_manually('touche21')

    dataset_name = 'argsme/2020-04-01/touche-2021-task-1'
    dataset = get_dataset(f'irds:{dataset_name}')

    # get qrels and docno of documents, which have a relevance judgment
    qrels = dataset.get_qrels()
    topics = dataset.get_topics()
    documents = dataset.get_corpus_iter()
    dc_cnt = 0
    for x in documents:
        dc_cnt += 1

    docnos_qrels = qrels['docno'].values
    nbr_of_qrels = len(docnos_qrels)
    nbr_of_documents = len(documents)
    print(f"Number of qrels: {nbr_of_qrels}")
    print(f"Number of documents: {dc_cnt}")

    docnos_set = set()

    def generate_filter_qrels() : # idea of filtering, negative that the size of the index is massively reduced
        for d in dataset.get_corpus_iter() :
            if (d['docno'] in docnos_qrels) : # qrel filtering happens on side of the participants
                if not isinstance(d['premises_texts'],str):
                    logger.warning(f"Document {d['docno']} has no premises_texts")
            yield {'docno' : d['docno'] , 'text' : d['premises_texts']}


    def generate_filter_minimal_length() :
        for d in dataset.get_corpus_iter() :
            if len(d['premises_texts'])  >= 10 and d['docno'] not in docnos_set:
                docnos_set.add(d['docno'])
                if d['topic']  != d['conclusion']:
                    mewo=1
                yield {'docno' : d['docno'] , 'text' : d['premises_texts'], 'title': d['topic']}
            #else:
             #   logger.warning(f"Document {d['docno']} has less than 10 characters")

    if s.dataset_index_dir.exists():
        delete_directory(s.dataset_index_dir)

    indexer = pt.terrier.IterDictIndexer(
        str(s.dataset_index_dir), meta={'docno': 100, 'text': 4096, 'title': 200}
        )
    indexer.index(generate_filter_minimal_length())


'''
dirichletLM = BatchRetrieve(str(index_dir.absolute()), wmodel='DirichletLM')

experiment = AxiomaticExperiment(
    retrieval_systems=[dirichletLM],
    topics=dataset.get_topics(),
    qrels=dataset.get_qrels(),
    index=index_dir,
    axioms=[],
    cache_dir=cache_dir,
    depth=5,
    filter_by_qrels=False,
)
with open(Path(__file__).parent / 'test.txt', 'w') as t:
    t.write(experiment.preferences.to_string())


dataset = pt.get_dataset('trec-deep-learning-docs')
a =  dataset.get_corpus_iter()

mewo = 1
for x in a:
    print(x)

bm25 = pt.terrier.Retriever.from_dataset(dataset, 'terrier_stemmed', wmodel='BM25')
dph = pt.terrier.Retriever.from_dataset(dataset, 'terrier_stemmed', wmodel='DPH')
pt.Experiment(
    [bm25, dph],
    dataset.get_topics(),
    dataset.get_qrels(),
    eval_metrics=['map']
)
'''