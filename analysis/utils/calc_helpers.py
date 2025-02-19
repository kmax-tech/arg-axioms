from collections import defaultdict
from analysis.utils.keys import *


class GetMetricDictOfDataframe():
    def __init__(self,df,metric_list,axioms_must_display=None):
        self.df = df
        self.metric_list = metric_list
        self.axioms_must_display = []
        if axioms_must_display is not None:
            self.axioms_must_display = axioms_must_display # axioms which has be displayed, e.g the new introduced ones

        self.data_dict = self.format_dataframe()

    def sort_dataframe(self,metric):
        df_filtered_sorted = self.df.sort_values(by=[metric] , ascending=False)
        df_filtered_sorted = df_filtered_sorted.reset_index(drop=True)
        return df_filtered_sorted

    def format_dataframe(self):
        data_to_return_dict = dict()

        for metric in self.metric_list:
            metric_dict = defaultdict(list)
            metric_name = metric[NAME_KEY]
            metric_significance = metric[SIGNIFICANT_KEY]

            sorted_dataframe = self.sort_dataframe(metric[NAME_KEY])

            axioms_to_display = sorted_dataframe['name'].tolist()
            if not len(self.axioms_must_display) == 0:
                best_axioms_old = [x for x in axioms_to_display if x not in self.axioms_must_display]
                best_axioms_old = best_axioms_old[:3] # only take the top 3 convential ones

                axioms_to_display = self.axioms_must_display + best_axioms_old

            for i,row in sorted_dataframe.iterrows():
                axiom = row['name']
                value = row[metric_name]
                value = str(round(value,3))

                if row[metric_significance] == True:
                    value = value + "$^{\dag}$" # dagger for significance


                metric_dict[AXIOM_KEY].append(str(axiom))
                metric_dict[SCORE_KEY].append(str(value))
                metric_dict[SCORE_KEY_RAW].append(row[metric_name])
                metric_dict[RANK_COLUMN_KEY].append(str(i + 1))
                metric_dict[RANK_COLUMN_KEY_RAW].append(i + 1)

            metric_dict[AXIOMS_TO_DISPLAY_KEY] = axioms_to_display
            data_to_return_dict[metric_name] = metric_dict
        return data_to_return_dict
