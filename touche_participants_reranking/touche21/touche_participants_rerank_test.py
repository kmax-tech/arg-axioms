import settings as s
import pyterrier as pt
import ir_measures
from pyterrier.pipelines import Experiment
import utils.repair_result_dataframe as rd
from pyterrier.io import read_results
import utils.get_datafeatures_from_datasets as gdf
if not pt.java.started() :
    pt.java.init()

if __name__ == '__main__':
    df_list = read_results('/Users/max/projects/axiomatic-reranking/touche21_participants_reranking_data/best_runs/Asterix/output-deduplicated-with-copycat/run5.txt')
    mewo = 1

    dataset = pt.datasets.get_dataset(f'irds:{s.dataset}')
    filtered_df = rd.adjust_retrieval_results_dataframe(df_list)
    bm25 = pt.terrier.Retriever(str(s.dataset_index_dir) , wmodel="BM25")
    result_transform = pt.Transformer().from_df(filtered_df)

    experiment = Experiment(
        retr_systems=[result_transform , bm25] ,
        topics=gdf.get_dataset_queries() ,
        qrels=gdf.get_qrels() ,
        eval_metrics=[ir_measures.nDCG(judged_only=True)@5] ,
        names=['test' , 'bm25'] ,
        baseline=0 ,
        correction="bonferroni" ,
        verbose=True ,
    )
    score_to_use = experiment.loc[experiment['name'] == 'test' ,'nDCG(judged_only=True)@5'].values[0]
    mewo = 1