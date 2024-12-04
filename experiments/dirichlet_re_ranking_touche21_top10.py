import pyterrier as pt
if not pt.java.started() :
    pt.java.init()
import matplotlib.pyplot as plt

import utils.repair_result_dataframe as rd
import ir_measures
import settings as s
import utils.get_datafeatures_from_datasets as gdf
from axioms.axioms_names import arg_axiom_list,arg_axiom_name_list,old_axioms,old_axioms_names
from experiments.utils.reranking_skeleton import re_rank_argu_axioms
import utils.save_runs as sr

if __name__ == '__main__':
    s.set_data_manually('touche21')

    experiment_name = 'dirichletlm-reranking-touche21-top10'

    indexref = pt.IndexRef.of(str(s.dataset_index_dir))
    index = pt.IndexFactory.of(indexref)
    stat = index.getCollectionStatistics()

    rerank_nbr = 10 # number of documents to rerank

    queries = gdf.get_dataset_queries()

    dirichletLM = pt.terrier.Retriever(str(s.dataset_index_dir) , wmodel="DirichletLM")
    res_df = dirichletLM.transform(queries)

    res_df = rd.cut_retrieval_results_top_n(res_df, rerank_nbr)

    rd.adjust_retrieval_results_dataframe_repair_get_missing(res_df,rerank_nbr) # fix missing judgements
    rd.set_retrieval_results_qrels_top_n(res_df, rerank_nbr) # Fix available qrels for this point

    metrics = [ir_measures.nDCG(judged_only=True)@5,ir_measures.nDCG(judged_only=True)@10,ir_measures.nDCG()@5,ir_measures.nDCG()@10]
    metric_names = ['nDCG(judged_only=True)@5','nDCG(judged_only=True)@10','nDCG@5','nDCG@10']

    axioms = arg_axiom_list + old_axioms
    axioms_names = arg_axiom_name_list + old_axioms_names

    experiment = re_rank_argu_axioms('DirichletLM',
                                     res_df,
                                     axioms=axioms,
                                     axioms_names=axioms_names,
                                     metrics=metrics,
                                     metrics_names=metric_names,
                                     rerank_nbr=rerank_nbr)

    sr.save_runs(experiment, experiment_name)
    # post processing
    df = experiment[["name"] + metric_names].copy()
    df[metric_names] = df[metric_names].round(3)

    # Create a figure and axis
    fig , ax = plt.subplots(figsize=(11.69 , 8.27))  # A4 landscape size in inches

    # Hide axes
    ax.axis('off')

    # Create a table and add it to the axes
    table = ax.table(cellText=df.values , colLabels=df.columns , cellLoc='center' , loc='center')

    # Adjust column width based on text
    for i , col in enumerate(df.columns) :
        max_length = max(df[col].astype(str).map(len).max() , len(col))  # Get max length of column
        table.auto_set_column_width(i)  # Automatically adjust width based on content
        table.scale(max_length / 5 , 1)  # Scale the width

    # Increase row height by scaling the height
    row_height = 2  # Adjust this value to increase/decrease height
    table.scale(1 , row_height)  # Scale width by 1 (unchanged) and height by row_height

    # Set the table font size
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Adjust layout
    plt.subplots_adjust(left=0.2 , right=0.8 , top=0.8 , bottom=0.2)
    # Save as PDF (landscape)
    plt.savefig(f'{experiment_name}.pdf' , format='pdf' , bbox_inches='tight')
    plt.show()

    # Create a figure and axis
    mewo = 3