import sys

from utils.get_datafeatures_from_datasets import get_qrels,get_dataset_queries,update_dataset_llm_qrels
from llm_prompting_qrels.llm_control import query_llm
import pandas as pd
import pyterrier as pt
if not pt.java.started() :
    pt.java.init()
import settings as s
from loguru import logger
import multi_experiments.multiproc as mc
def repair_touche_run(df):
    queries_df = get_dataset_queries()

    df_merged = pd.merge(df, queries_df, on='qid')

    indexref = pt.IndexRef.of(str(s.dataset_index_dir))
    index = pt.IndexFactory.of(indexref)
    meta = index.getMetaIndex()

    collect_docids  =  []
    for docno in df_merged['docno']:
        docid = meta.getDocument('docno' , docno)
        if docid != -1:
            docno_index = meta.getItem("docno" , docid)
            assert docno == docno_index
        collect_docids.append(docid)
    df_merged['docid'] = collect_docids

    df_merged_only_docs_in_index = df_merged[df_merged['docid'] != -1] # filter out invalid docnos

    if len(df_merged_only_docs_in_index) == 0:
        logger.error(f"No documents in Index found")
        return None


    diff = len(df_merged) - len(df_merged_only_docs_in_index)
    logger.info(f"Number of documents filtered out from Touche Run {diff}")
    qids = df_merged_only_docs_in_index['qid'].unique()
    sub_frames = []
    for x in qids :
        repaired = rank_repair(df_merged_only_docs_in_index , x)
        sub_frames.append(repaired)
    dataframe_participant = pd.concat(sub_frames)

    return dataframe_participant[['qid','docid', 'docno','rank','score','query']]

# trim the retrieval results to the top n, used if the qrels are not complete and need to be calculated
def cut_retrieval_results_top_n(df,cut_off=20):
    filtered_df = df[(df['rank'] >= 0) & (df['rank'] < (cut_off))]
    return filtered_df

# repair missing, ranks, caused through filtering
def rank_repair(df,qid_to_update):
    df_qid = df[df['qid'] == qid_to_update]
    reference = df_qid.copy()

    # Keep only the row with the highest score for each rank
    df_filtered = df_qid.loc[df_qid.groupby('rank')['score'].idxmax()]
    df_qid_sorted = df_filtered.sort_values(by='score', ascending=False).reset_index(drop=True)
    df_qid_sorted['rank'] = df_qid_sorted.index

    if len(reference) != len(df_qid_sorted):
        logger.info(f"Rank repair for qid {qid_to_update} has been performed, removed {len(reference) - len(df_qid_sorted)} documents")


    return df_qid_sorted

def adjust_drop_missing_docnos(df):
    docnos_qrels = get_qrels(just_human=True)[['qid','docno']]

    df_sorted = df.sort_values(by=['qid' , 'rank'] , ascending=[True , True])

    df_filtered = df_sorted.merge(docnos_qrels , on=['qid' , 'docno'] , how='left' , indicator=True)
    df_filtered = df_filtered[df_filtered['_merge'] == 'both'].drop(columns=['_merge'])
    qids = df_filtered['qid'].unique()
    sub_frames = []
    for x in qids:
        repaired = rank_repair(df_filtered, x)
        sub_frames.append(repaired)
    return pd.concat(sub_frames)

def update_dataset_llm_qrels_helper(data_llm):
    relevance = data_llm['relevance_llm']
    quality = data_llm['quality_llm']
    docno = data_llm['docno']
    qid = data_llm['qid']

    if (relevance is None) or (quality is None):
        logger.error(f"Document {docno} has no relevance or quality")
        sys.exit(1)
    if relevance not in [-2,0,1,2]:
        logger.error(f"Document {docno} has invalid relevance {relevance}")
        sys.exit(1)
    if quality not in [0,1,2]:
        logger.error(f"Document {docno} has invalid quality {quality}")
        sys.exit(1)

    update_dataset_llm_qrels(data_llm)
    logger.info(f"Annotated document {docno} with relevance {relevance} and quality {quality}")

def adjust_retrieval_results_dataframe_repair_get_missing(df,cut_off=20):
    filtered_df = cut_retrieval_results_top_n(df,cut_off)

    qrel_df = get_qrels(ignore_qrels_to_use=True)
    queries = get_dataset_queries()
    docnos_qrels = qrel_df['docno'].values

    df_combined = pd.merge(filtered_df , qrel_df , on=['qid' , 'docno'] , how='left' , indicator=True)
    df_unique_to_df1 = df_combined[df_combined['_merge'] == 'left_only']

    indexref = pt.IndexRef.of(str(s.dataset_index_dir))
    index = pt.IndexFactory.of(indexref)
    # inv = index.getInvertedIndex()
    meta = index.getMetaIndex()
    logger.info(f"Number of documents to annotate {len(df_unique_to_df1)}")
    if len(df_unique_to_df1) == 0 :
        return

    data_all_instances_to_annotate = []

    for i,x in df_unique_to_df1.iterrows():
        docid = x['docid']
        qid = x['qid']
        docno = x['docno']
        index_docno = meta.getItem("docno" , docid)
        assert docno == index_docno

        document = meta.getItem("text" , docid)
        title = meta.getItem("title" , docid)

        topic = queries[queries['qid'] == qid]['query'].values[0]

        data_llm_dict = { 'topic' : topic , 'document' : document,'title' : title, 'docno' : docno, 'qid' : qid}
        data_all_instances_to_annotate.append(data_llm_dict)



    if s.NBR_PROCESSES:

        def split_into_chunks(lst, chunk_size=20) :
            return [lst[i :i + chunk_size] for i in range(0, len(lst), chunk_size)]

        # general splitting to entries can be saved ealier in case of exceptions
        general_split = split_into_chunks(data_all_instances_to_annotate, 20)

        for data_big_chunk in general_split:
            mccalc = mc.MCCalc()
            mp_calc_method = mc.ExperimentWorker(query_llm)
            data_split = mccalc.split_text(data_big_chunk)
            all_data_llm_list_mp, error = mccalc.run_mult(mp_calc_method.run, data_split)
            if 1 in error:
                logger.error(f"Error in LLM API: {error}")
                sys.exit(1)
            all_data_llm_list = []
            for x in all_data_llm_list_mp:
                all_data_llm_list.extend(x)

            for data_llm in all_data_llm_list:
                update_dataset_llm_qrels_helper(data_llm)

    else:
        for data_llm_dict in data_all_instances_to_annotate:
            data_llm = query_llm(data_llm_dict)
            update_dataset_llm_qrels_helper(data_llm)


# adjust the documents, which are in the index
def set_retrieval_results_qrels_top_n(dataframes, top):
    considered_documents = []
    if not isinstance(dataframes, list):
        dataframes = [dataframes]
    for df in dataframes:
        filtered_df = cut_retrieval_results_top_n(df, top)  # adjust missing nbrs of document
        filtered_df_relevance = filtered_df[['qid','docno']]
        considered_documents.append(filtered_df_relevance)

    all_relevant_df = pd.concat(considered_documents)
    all_relevant_df = all_relevant_df.drop_duplicates()
    qrel_df_human = get_qrels(just_human=True,ignore_qrels_to_use=True)
    qrel_df = get_qrels(ignore_qrels_to_use=True)

    filtered_df = qrel_df.merge(all_relevant_df, on=['qid', 'docno'], how='left', indicator=True)
    filtered_df = filtered_df[filtered_df['_merge'] == 'both'].drop(columns=['_merge'])
    assert len(filtered_df) == len(all_relevant_df)
    assert filtered_df.shape != all_relevant_df.shape

    # merged_df_test = filtered_df.merge(qrel_df_human, how='inner')

    merged_df = pd.concat([filtered_df, qrel_df_human])
    merged_df = merged_df.drop_duplicates()


    s.QRELS_TO_USE = merged_df
