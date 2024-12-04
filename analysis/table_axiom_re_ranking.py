import pandas as pd
from collections import defaultdict
from axioms.axioms_names import arg_axiom_list,arg_axiom_name_list,old_axioms,old_axioms_names
from utils.save_runs import load_runs
from axioms.axioms_translate_dict import arg_translate_dict
import analysis.utils.latex_helper as lh
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
axioms_must_display = arg_axiom_name_list  + ["DirichletLM"]

final_row_order =[axiom_key, ndcg5[name_key], ndcg10[name_key],  ndcg5[name_key]+rank_column_key, ndcg10[name_key]+rank_column_key]



class GetMetricsToTable():
    ''''Create Table, which can be printed'''

    def __init__(self,dataframes):
        self.dataframes = dataframes
        self.touche_metric_dict_list = []
        self.axioms_to_display = axioms_must_display
        self.nbr_of_rows = None

        self.final_table_columns_list = []

        self.prepare_table()
        self.get_columns_table()
        self.pretty_print_table()



    def create_empty_columns(self):
        return ['' for x in range(self.nbr_of_rows)]

    def prepare_table(self):
        for dataframe in self.dataframes:
            data_dict = GetMetricDictOfDataframe(dataframe,metric_list,axioms_must_display).data_dict
            for metric in metric_list:
                metric_name = metric[name_key]
                axioms_to_display = data_dict[metric_name][axioms_to_display_key]
                self.axioms_to_display += axioms_to_display
            self.touche_metric_dict_list.append(data_dict)
        self.axioms_to_display = list(set(self.axioms_to_display)) # clean out duplicates

    def get_columns_table(self):
        '''combine metrics from multiple dataframes'''
        for metric_i,data_dict in enumerate(self.touche_metric_dict_list):
            data_dict_touche = defaultdict(list)

            axioms_to_display_order_tmp =  data_dict[ndcg5[name_key]][axiom_key]
            axioms_to_display_order = [x for x in axioms_to_display_order_tmp if x in self.axioms_to_display]
            data_dict_touche[axiom_key] = [arg_translate_dict(x) for x in axioms_to_display_order]
            for metric in metric_list:
                metric_name = metric[name_key]
                metric_dict = data_dict[metric_name]

                metric_rank_column_key = metric_name + rank_column_key

                for axiom_order in axioms_to_display_order:
                    for i,axiom in enumerate(metric_dict[axiom_key]):
                        if axiom == axiom_order:
                            data_dict_touche[metric_rank_column_key].append(metric_dict[rank_column_key][i])
                            data_dict_touche[metric_name].append(metric_dict[score_key][i])

            for key in final_row_order:
                key_data_list = data_dict_touche[key]
                self.final_table_columns_list.append([key,key_data_list])

            self.nbr_of_rows = max([len(vals) for key, vals in self.final_table_columns_list])
            if metric_i < len(self.touche_metric_dict_list) - 1:
                self.final_table_columns_list.append(['emp',['']])
                self.final_table_columns_list.append(['emp',['']])

    def adjust_table_length(self):
        for key,value in self.final_table_columns_list:
            if len(value) != self.nbr_of_rows:
                diff = self.nbr_of_rows - len(value)
                value += ['' for x in range(diff)]

    def pretty_print_table(self):
        self.adjust_table_length()
        table_process = lh.columns_to_rows(self.final_table_columns_list)
        table_process = lh.latex_row_command(table_process, 'DirichletLM',columns_to_ignore=[5,6,7,8,9,10])
        table_process = lh.latex_row_command(table_process, 'DirichletLM', ident=7, columns_to_ignore=[1,2,3,4])

        table_process = lh.latex_mark_highest_column(table_process,index_to_ignore=[0])

        table_width_adjusted = lh.fill_rows(table_process[1:])
        table = PrettyTable()
        #table.field_names = table_raw[0]
        table.add_rows(
                table_process[0:1] +
                table_width_adjusted

        )
        latex_table = table.get_latex_string()
        print(latex_table)







if __name__ == '__main__':
    touche21_top5 = load_runs('dirichletlm-reranking-touche21-top5')
    dataframe_touche20 = load_runs('dirichletlm-reranking-touche20-top10')
    dataframe_touche21 = load_runs('dirichletlm-reranking-touche21-top10')
    # dataframe_touche21_only_judged = load_runs('DirichletLM-Reranking-Only-Judged')
    # dataframe_touche21_metric_only_judged = load_runs('DirichletLM-Reranking-Only-Judged')
    x = GetMetricsToTable([dataframe_touche20,dataframe_touche21])
