import copy

import pandas as pd
from axioms.axioms_names import arg_axiom_list,arg_axiom_name_list,old_axioms,old_axioms_names
from utils.save_runs import load_runs
from axioms.axioms_names import arg_axiom_list,arg_axiom_name_list,old_axioms,old_axioms_names
from axioms.axioms_translate_dict import arg_translate_dict
import analysis.utils.latex_helper as lh
from collections import defaultdict
from prettytable import PrettyTable
from analysis.utils.keys import *
from analysis.utils.calc_helpers import GetMetricDictOfDataframe

ndcg5 = {
name_key : 'nDCG(judged_only=True)@5', # column in the dataframe
significant_key : 'nDCG(judged_only=True)@5 reject', # column indicating significance
'display_name' : '@5', # new name of the column used for display
}

ndcg10 = {
name_key : 'nDCG(judged_only=True)@10', # column in the dataframe
significant_key : 'nDCG(judged_only=True)@10 reject', # column indicating significance
'display_name' : '@10', # new name of the column used for display
}

# contains entry for all displayed metrics
metric_list = [ndcg5,ndcg10]
axioms_to_display =['base'] + arg_axiom_name_list

non_exact_keys = ['QSenSim_max_sbert','QSenSim_mean_sbert','QArgSim_max_sbert_full_document','QArgSim_mean_sbert_full_document']
exact_keys = ['QSenSim_max_exact_sbert','QSenSim_mean_exact_sbert','QArgSim_max_exact_sbert_full_document','QArgSim_mean_exact_sbert_full_document']

# only show exact terms
final_row_order = ['name','base_score'] + [item for x in non_exact_keys for item in (f'{arg_translate_dict(x)}_score', f'{arg_translate_dict(x)}_pos')]
#final_row_order = ['name','base_score'] + [item for x in exact_keys for item in (f'{arg_translate_dict(x)}_score', f'{arg_translate_dict(x)}_pos')]

# use it for calculation
class ToucheRerankingAnalysis():
    def __init__(self,data_list_participants,metric_to_use):
        self.data_dicts_participants = dict()
        self.metric_to_use = metric_to_use
        self.final_table_columns_list = []

        self.preprocess_dataframes(data_list_participants)
        self.get_scoring_table()
        self.pretty_print_table()

    # loop through all groups and all  and combine data to a multidict
    def preprocess_dataframes(self,data_list_participants):
        for group,df in data_list_participants: # collect corresponding dfs for each group
            metric_datas = GetMetricDictOfDataframe(df,metric_list,None).data_dict
            metric_data = metric_datas[self.metric_to_use]


            change_axioms_to_display = metric_data[axioms_to_display_key]
            change_axioms_to_display = ["base" if item == group else item for item in change_axioms_to_display]
            metric_data[axioms_to_display_key] = change_axioms_to_display

            change_axioms = metric_data[axiom_key]
            change_axioms = ["base" if item == group else item for item in change_axioms]
            metric_data[axiom_key] = change_axioms

            self.data_dicts_participants[group] = metric_data


    # sort the entries in the list according to a specified criteria
    def create_group_axiom_info_dict(self, axiom):
        groups_data_list = []
        for group, group_data in self.data_dicts_participants.items():
            i = group_data[axiom_key].index(axiom)
            score = group_data[score_key][i]
            score_raw = group_data[score_raw_key][i]

            data_dict = dict()
            data_dict[group_key] = group
            data_dict[score_key] = score
            data_dict[score_raw_key] = score_raw
            groups_data_list.append(data_dict)

        groups_scores_dict = {x[group_key]: x for x in groups_data_list}
        return groups_scores_dict

    def order_dict_by_score(self,groups_scores_dict):
        group_list = [(x,y) for x,y in groups_scores_dict.items()]
        group_sorted = sorted(group_list, key=lambda x: x[1][score_key], reverse=True)
        return group_sorted
    def create_score_dict(self, data_dict):
        return {x : y[score_key] for x,y in self.order_dict_by_score(data_dict)}

    def create_rank_dict(self, data_dict):
        return {x : i + 1 for i,(x,y) in enumerate(self.order_dict_by_score(data_dict))}

    # calculate the differences in ranks between if axioms are used
    def calculate_differences_in_scores(self,axiom):
        base_group_dict= self.create_group_axiom_info_dict('base')
        base_rank_dict = self.create_rank_dict(base_group_dict)

        axiom_group_dict = self.create_group_axiom_info_dict(axiom)

        difference_dict = dict()

        for participant in base_group_dict.keys():
            participant_axiom_score = axiom_group_dict[participant][score_key]
            participant_axiom_score_raw = axiom_group_dict[participant][score_raw_key]

            dict_to_rank = copy.deepcopy(base_group_dict)
            dict_to_rank[participant][score_key] = participant_axiom_score
            dict_to_rank[participant][score_raw_key] = participant_axiom_score_raw

            axiom_ranked_dict = self.create_rank_dict(dict_to_rank)

            base_rank = base_rank_dict[participant]
            axiom_rank = axiom_ranked_dict[participant]

            difference = base_rank - axiom_rank
            difference_dict[participant] = (axiom_rank,difference)
        return difference_dict

    def get_scoring_table(self) :
        base_dict = self.create_group_axiom_info_dict('base')
        participants_names = [x for x,y in self.order_dict_by_score(base_dict)]

        participants_names_display = [arg_translate_dict(x) for x in participants_names]
        self.final_table_columns_list.append(('name',participants_names_display))

        for axiom in axioms_to_display:
            axiom_score_list = []
            axiom_position_changes_list = []

            axiom_group_dict = self.create_group_axiom_info_dict(axiom)
            axiom_scores_dict = self.create_score_dict(axiom_group_dict)
            axiom_differences_dict = self.calculate_differences_in_scores(axiom)

            for participant in participants_names:
                axiom_score_list.append(axiom_scores_dict[participant])
                rank_participant,difference_participant = axiom_differences_dict[participant]
                positional_data = f"{rank_participant} ({difference_participant})"
                axiom_position_changes_list.append(positional_data)

            # set new keys
            axiom_score = f"{arg_translate_dict(axiom)}_score"
            axiom_position_changes = f"{arg_translate_dict(axiom)}_pos"
            self.final_table_columns_list.append((axiom_score,axiom_score_list))
            self.final_table_columns_list.append((axiom_position_changes,axiom_position_changes_list))

        # get the final_tabLe_columns_list into specified line
        final_order_pre = []
        for x in final_row_order:
            for key,value in self.final_table_columns_list:
                if key == x:
                    final_order_pre.append((key,value))
        self.final_table_columns_list = final_order_pre



    def pretty_print_table(self):
        table_raw = lh.columns_to_rows(self.final_table_columns_list)
        table_head = table_raw[:1]
        table_body = table_raw[1:]
        table_body = lh.latex_mark_highest_column(table_body,index_to_ignore=[0,1])
        table_body = lh.latex_row_command(table_body, 'DirichletLM')
        table_width_adjusted = lh.fill_rows(table_body)
        table = PrettyTable()
        table.add_rows(
                table_head +
                table_width_adjusted

        )
        latex_table = table.get_latex_string()
        print(latex_table)




if __name__ == '__main__':

    a = load_runs('touche-21-reranking')
    print(metric_list[0][name_key])
    ToucheRerankingAnalysis(a,metric_list[0][name_key])
    print(metric_list[1][name_key])
    ToucheRerankingAnalysis(a,metric_list[1][name_key])








