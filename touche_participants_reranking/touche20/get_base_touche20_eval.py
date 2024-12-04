from utils import group_name_helpers as gnh
import settings as s
from touche_participants_reranking.evaluate_single_touche_run_base import evaluate_single_touche_run_ndcg
import utils.repair_result_dataframe as rd
from utils.save_runs import save_runs
from functools import partial

# calculate base performance for the involved touche runs, use it to determine the best run


if __name__ == "__main__":

    s.set_data_manually('touche20')
    s.TOUCHE__DIR = '/Users/max/projects/axiomatic-reranking/_touche20_participant_data'

    experiment_name = 'touche-20-base_evaluation'
    ndcg_nbr = 5

    # use of the participants, which perform better than the baseline

    # first get all runs from the participants, every bit of data is pooled
    # this is necessary for getting a united pool of qrels
    score_func_touche20 = partial(evaluate_single_touche_run_ndcg , ndcg_nbr)

    participants = gnh.get_all_group_participants(touche_20_data_dir)

    data_frames_for_participants = []
    for name in participants:
        runs = gnh.get_runs_from_participants_touche20(touche_20_data_dir,name)
        for run in runs:
            rd.adjust_retrieval_results_dataframe_repair_get_missing(run[1], ndcg_nbr)
            data_frames_for_participants.append(run[1])

    rd.set_retrieval_results_qrels_top_n(data_frames_for_participants, ndcg_nbr)

    # do the final calculations for the involved data
    data_collection = []
    for name in participants:
        data = gnh.get_best_run_for_participant(name, score_func_touche20)
        data_collection.append(data)

    save_runs(data_collection, experiment_name)
