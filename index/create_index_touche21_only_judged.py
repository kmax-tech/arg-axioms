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
    s.set_data_manually('touche21-only-judged')

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


    def generate_filter_qrels() :
        for d in dataset.get_corpus_iter() :
            if len(d['premises_texts'])  >= 10 and d['docno'] not in docnos_set:
                if (d['docno'] in docnos_qrels) :
                    docnos_set.add(d['docno'])
                    yield {'docno' : d['docno'] , 'text' : d['premises_texts'], 'title': d['topic']}
                #else:
             #   logger.warning(f"Document {d['docno']} has less than 10 characters")

    if s.dataset_index_dir.exists():
        delete_directory(s.dataset_index_dir)

    indexer = pt.terrier.IterDictIndexer(
        str(s.dataset_index_dir), meta={'docno': 100, 'text': 4096, 'title': 200}
        )
    indexer.index(generate_filter_qrels())

