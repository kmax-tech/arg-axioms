from collections import defaultdict
from analysis.utils.keys import *


class GetMetricDictOfDataframe():
    def __init__(self,df,metric_list,axioms_must_display=None):
        self.df = df
        self.metric_list = metric_list
        self.axioms_must_display = []
        if axioms_must_display is not None:
            self.axioms_must_display = axioms_must_display

        self.data_dict = self.format_dataframe()

    def sort_dataframe(self,metric):
        df_filtered_sorted = self.df.sort_values(by=[metric] , ascending=False)
        df_filtered_sorted = df_filtered_sorted.reset_index(drop=True)
        return df_filtered_sorted

    def format_dataframe(self):
        data_to_return_dict = dict()

        for metric in self.metric_list:
            metric_dict = defaultdict(list)
            metric_name = metric[name_key]
            metric_significance = metric[significant_key]

            sorted_dataframe = self.sort_dataframe(metric[name_key])

            axioms_to_display = sorted_dataframe['name'].tolist()
            if not len(self.axioms_must_display) == 0:
                best_axioms_old = [x for x in axioms_to_display if x not in self.axioms_must_display]
                best_axioms_old = best_axioms_old[:3]

                axioms_to_display = self.axioms_must_display + best_axioms_old

            for i,row in sorted_dataframe.iterrows():
                axiom = row['name']
                value = row[metric_name]
                value = str(round(value,3))

                if row[metric_significance] == True:
                    value = value + "$^{\dag}$" # dagger for significance


                metric_dict[axiom_key].append(str(axiom))
                metric_dict[score_key].append(str(value))
                metric_dict[score_raw_key].append(row[metric_name])
                metric_dict[rank_column_key].append(str(i + 1))
                metric_dict[rank_column_raw_key].append(i + 1)

            metric_dict[axioms_to_display_key] = axioms_to_display
            data_to_return_dict[metric_name] = metric_dict
        return data_to_return_dict
