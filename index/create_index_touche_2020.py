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
    s.set_data_manually('touche20')
    dataset_name = 'beir/webis-touche2020/v2'
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

    drop_out_cnt = 0
    indexed_cnt = 0
    total_cnt = 0

    def generate_filter_qrels() : # idea of filtering, negative that the size of the index is massively reduced
        for d in dataset.get_corpus_iter() :
            if (d['docno'] in docnos_qrels) : # qrel filtering happens on side of the participants
                if not isinstance(d['text'],str):
                    logger.warning(f"Document {d['docno']} has no premises_texts")
            yield {'docno' : d['docno'] , 'text' : d['text']}


    def generate_filter_minimal_length() :
        global docnos_set
        global drop_out_cnt
        global indexed_cnt
        global total_cnt
        for d in dataset.get_corpus_iter() :
            total_cnt += 1
            if len(d['text'])  >= 10 and d['docno'] not in docnos_set:
                docnos_set.add(d['docno'])
                indexed_cnt += 1
                yield {'docno' : d['docno'] , 'text' : d['text'], 'title': d['title']}
            else:
                drop_out_cnt += 1
             #   logger.warning(f"Document {d['docno']} has less than 10 characters")

    if s.dataset_index_dir.exists():
        delete_directory(s.dataset_index_dir)

    indexer = pt.terrier.IterDictIndexer(
        str(s.dataset_index_dir), meta={'docno': 100, 'text': 4096, 'title': 200}
        )
    indexer.index(generate_filter_minimal_length())
    print(f"Indexing finished. {indexed_cnt} documents were indexed.")
    print(f"Indexing finished. {drop_out_cnt} documents were dropped out.")
    print(f"Indexing finished. {total_cnt} documents were processed.")
    print('docnos set',len(docnos_set))


