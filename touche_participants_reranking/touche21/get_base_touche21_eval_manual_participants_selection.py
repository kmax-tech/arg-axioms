from utils import group_name_helpers as gnh
import settings as s
from touche_participants_reranking.evaluate_single_touche_run_base import evaluate_single_touche_run_ndcg
import utils.repair_result_dataframe as rd
from utils.save_runs import save_runs
from functools import partial
from slugify import slugify
# calculate base performance for the involved touche runs, use it to determine the best run


if __name__ == "__main__":

    # specify the use of touche21
    s.set_data_manually('touche21')
    s.TOUCHE_DIR = '/Users/max/projects/axiomatic-reranking/_touche21_participant_data'
    experiment_name = 'touche-21-base_evaluation_manual_participants_selection'
    ndcg_nbr = 5

    # use of the participants, which perform better than the baseline
    participants = ['Elrond',
             'Pippin Took',
             'Robin Hood',
             'Asterix',
             'Dread Pirate Roberts',
             'Skeletor',
             'Luke Skywalker',
             'Shanks',
             'Heimdall',
             'Athos',
             'Goemon Ishikawa',
             'Jean Pierre Polnareff',
             'Baseline-Dirichlet']
    participants = [slugify(p).lower() for p in participants]

    # first get all runs from the participants, every bit of data is pooled
    # this is necessary for getting a united pool of qrels
    score_func_touche21 = partial(evaluate_single_touche_run_ndcg , ndcg_nbr)

    #participants = gnh.get_all_group_participants()

    data_frames_for_participants = []
    for name in participants:
        runs = gnh.get_runs_from_participants_touche21(name)
        for run in runs:
            rd.adjust_retrieval_results_dataframe_repair_get_missing(run[1], ndcg_nbr)
            data_frames_for_participants.append(run[1])

    rd.set_retrieval_results_qrels_top_n(data_frames_for_participants, ndcg_nbr)

    #gnh.get_best_run_for_participant('elrond', get_run_func=gnh.get_runs_from_participants_touche21,
     #                                experiment_func=score_func_touche21)
    # do the final calculations for the involved data
    data_collection = []
    for name in participants:
        data = gnh.get_best_run_for_participant(name,get_run_func=gnh.get_runs_from_participants_touche21, experiment_func=score_func_touche21)
        data_collection.append((name,data))

    save_runs(data_collection, experiment_name)
