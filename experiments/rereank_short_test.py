import settings as s
import pyterrier as pt
import ir_measures
import utils.repair_result_dataframe as rd
if not pt.java.started() :
    pt.java.init()
from pyterrier.batchretrieve import BatchRetrieve
from pyterrier.pipelines import Experiment
from ir_axioms.modules.pivot import MiddlePivotSelection
from ir_axioms.backend.pyterrier.transformers import KwikSortReranker
from axioms.qargsim_sbert import QArgSim_mean_exact_sbert,QArgSim_mean_exact_sbert_full_document
from axioms.qsensim_sbert import QSenSim_max_sbert
from axioms.stmc_sbert import STMC1_sbert
import pandas as pd



from axioms.qsensim_ada import QSenSim_max_exact_ada

if __name__ == "__main__":

    s.set_data_manually('touche21')
    dataset = pt.datasets.get_dataset(f'irds:{s.dataset}')

    topics = dataset.get_topics()
    qrels = dataset.get_qrels()
    topic_qrels = dataset.get_topicsqrels()
    indexref = pt.IndexRef.of(str(s.dataset_index_dir))

    nwo = 1

    dirichletLM = pt.terrier.Retriever(str(s.dataset_index_dir), wmodel="DirichletLM")
    bm25 = pt.terrier.Retriever(str(s.dataset_index_dir), wmodel="BM25")

    res_df = dirichletLM.transform(topics)

    filtered_df = res_df[(res_df['rank'] >= 0) & (res_df['rank'] <= 20)]

    # cut out underlying values
    filtered_df = rd.adjust_drop_missing_docnos(filtered_df)

    # If you only want to get the 'br' column
    result = filtered_df['docno'].to_list()
    resultids = filtered_df['docid'].to_list()

    index = pt.IndexFactory.of(indexref)
    # inv = index.getInvertedIndex()
    meta = index.getMetaIndex()

    # Get the number of documents
    num_documents = index.getCollectionStatistics()
    print(f"Number of documents: {num_documents}")
    filtered_documents = {
    s.docnos : [meta.getItem("docno",x) for x in resultids],
    s.texts : [meta.getItem("text",x) for x in resultids],
    s.TASK : "BATCH_EMBEDDING"
    }


    assert len(filtered_documents[s.texts]) == len(filtered_documents[s.docnos])

    result_transform = pt.Transformer().from_df(filtered_df)
    mewo = 1
    # send_data_to_socket(filtered_documents)



    def kwiksort(retr_sys , axioms) :
        return [retr_sys % 5 >> KwikSortReranker(
            axiom=axiom ,
            index=s.dataset_index_dir ,
            pivot_selection=MiddlePivotSelection() ,
            cache_dir=s.cache_dir
        ) ^ retr_sys
                for axiom in axioms]

    axioms = [STMC1_sbert(),QArgSim_mean_exact_sbert(),QArgSim_mean_exact_sbert_full_document(),QSenSim_max_sbert()]
    axioms_names = ['STMC1_sbert','QArgSim_mean_exact_sbert','QArgSim_mean_exact_sbert_full_document','QSenSim_max_sbert']
    rerank_dirichletLM = kwiksort(result_transform, axioms)


    experiment = Experiment(
        retr_systems=[dirichletLM,result_transform] + rerank_dirichletLM,
        topics=dataset.get_topics(),
        qrels=dataset.get_qrels(),
        eval_metrics=[ir_measures.nDCG(judged_only=True) @ 5],
        names=["DirichletLM",'DirichletLMFiltered'] + axioms_names,
        baseline=0,
        correction="bonferroni",
        verbose=True,
    )
    experiment.sort_values(by="nDCG(judged_only=True)@5", ascending=False, inplace=True)

    # Printing the dataframe
    print(experiment)
    mewo = 1






