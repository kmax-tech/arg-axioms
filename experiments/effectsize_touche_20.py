import pyterrier as pt
if not pt.java.started() :
    pt.java.init()
import matplotlib.pyplot as plt

import utils.repair_result_dataframe as rd
import ir_measures
import settings as s
import utils.get_datafeatures_from_datasets as gdf
from axioms.axioms_names import arg_axiom_list,arg_axiom_name_list,old_axioms,old_axioms_names
from experiments.utils.reranking_effectsize_skeleton import re_rank_argu_axioms_effectsize
import utils.save_runs as sr

if __name__ == '__main__':
    s.set_data_manually('touche20')

    experiment_name = 'dirichletlm-reranking-touche20-top10-effectsize'

    indexref = pt.IndexRef.of(str(s.dataset_index_dir))
    index = pt.IndexFactory.of(indexref)
    stat = index.getCollectionStatistics()

    rerank_nbr = 10  # number of documents to rerank

    queries = gdf.get_dataset_queries()

    dirichletLM = pt.terrier.Retriever(str(s.dataset_index_dir) , wmodel="DirichletLM")
    res_df = dirichletLM.transform(queries)

    res_df = rd.cut_retrieval_results_top_n(res_df, rerank_nbr)

    rd.adjust_retrieval_results_dataframe_repair_get_missing(res_df,rerank_nbr) # fix missing judgements
    rd.set_retrieval_results_qrels_top_n(res_df, rerank_nbr) # Fix available qrels for this point

    metrics = [ir_measures.nDCG(judged_only=True)@5,ir_measures.nDCG(judged_only=True)@10,ir_measures.nDCG()@5,ir_measures.nDCG()@10]
    metric_names = ['nDCG(judged_only=True)@5','nDCG(judged_only=True)@10','nDCG@5','nDCG@10']

    axioms = arg_axiom_list
    axioms_names = arg_axiom_name_list

    baseline = 'baseline-dirichlet'

    experiment = re_rank_argu_axioms_effectsize(baseline,
                                                res_df,
                                                axioms=axioms,
                                                axioms_names=axioms_names,
                                                metrics=metrics,
                                                metrics_names=metric_names,
                                                rerank_nbr=rerank_nbr)

    sr.save_runs(experiment, experiment_name)

