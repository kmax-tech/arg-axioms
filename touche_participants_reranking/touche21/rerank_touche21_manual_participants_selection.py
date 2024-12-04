import settings
from utils.save_runs import save_runs,load_runs
import utils.repair_result_dataframe as rd
from axioms.axioms_names import arg_axiom_list,arg_axiom_name_list,old_axioms,old_axioms_names
from experiments.utils.reranking_skeleton import re_rank_argu_axioms

import ir_measures

if __name__ == '__main__':

    settings.set_data_manually('touche21')
    # do reranking for touche21 participants, with the new involved axioms
    experiment_name = 'touche-21-reranking_manual_participants_selectio'

    base_run_data_list = load_runs('touche-21-base_evaluation_manual_participants_selection')
    rerank_nbr = 10 # number of documents to rerank

    best_runs = []
    for i,data in enumerate(base_run_data_list):
        group = data[0]
        best_run = data[3]
        rd.adjust_retrieval_results_dataframe_repair_get_missing(best_run, rerank_nbr)
        best_runs.append((group,best_run))

    # adjust the underlying qrels
    best_runs_adj = [x[1] for x in best_runs]
    rd.set_retrieval_results_qrels_top_n(best_runs_adj, rerank_nbr)

    metrics = [ir_measures.nDCG(judged_only=True) @ 5, ir_measures.nDCG(judged_only=True) @ 10, ir_measures.nDCG() @ 5,
               ir_measures.nDCG() @ 10]
    metric_names = ['nDCG(judged_only=True)@5', 'nDCG(judged_only=True)@10', 'nDCG@5', 'nDCG@10']

    axioms =  arg_axiom_list # + old_axioms
    axioms_names = arg_axiom_name_list  # arg_axiom_name_list #+ old_axioms_names

    re_ranked_data_participants = []
    for group, df in best_runs:
        experiment_group = re_rank_argu_axioms(group,
                                         df,
                                         axioms=axioms,
                                         axioms_names=axioms_names,
                                         metrics=metrics,
                                         metrics_names=metric_names,
                                         rerank_nbr=rerank_nbr)
        re_ranked_data_participants.append((group,experiment_group))

    save_runs(re_ranked_data_participants, experiment_name)