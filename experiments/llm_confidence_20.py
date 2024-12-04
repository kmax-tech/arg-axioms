import pandas as pd
from pyterrier.datasets import get_dataset
from pathlib import Path
import pickle
import pyterrier as pt
if not pt.java.started() :
    pt.java.init()
import utils.get_datafeatures_from_datasets as gdf
from llm_prompting_qrels.llm_control import query_llm
import settings as s

s.set_data_manually('touche20')
s.LLMS_TO_USE = [s.CLAUDE ,s.GEMINI,s.GPT]

dataset_name = "beir/webis-touche2020/v2"
dataset = get_dataset(f"irds:{dataset_name}")

qrels_df = dataset.get_qrels()
qrels_df = qrels_df.sort_values(by=['qid', 'docno'], ascending=True)
queries = gdf.get_dataset_queries()
indexref = pt.IndexRef.of(str(s.dataset_index_dir))
index = pt.IndexFactory.of(indexref)
meta = index.getMetaIndex()

storage_data_name = Path("correlation_data_qrels_llms_20.pkl")


if storage_data_name.exists():
    with open(storage_data_name, "rb") as pickle_file:
        evaluated_df = pickle.load(pickle_file)
else:
    evaluated_df = qrels_df.sample(frac=0.25, random_state=1)
    evaluated_df['relevance_llm'] = pd.NA
    evaluated_df['quality_llm'] = pd.NA

progress_count = 1
for i, row  in evaluated_df.iterrows():

    progress = progress_count / len(evaluated_df) * 100  # Calculate progress percentage
    progress_count += 1
    print(f"Progress: {progress:.2f}%")  # Print progress rounded to 2 decimal places

    relevance_llm = row['relevance_llm']
    quality_llm = row['quality_llm']
    if not (pd.isna(relevance_llm) or pd.isna(quality_llm)):
        continue

    qid = row['qid']
    docno = row['docno']
    docid = meta.getDocument('docno' , docno)
    if docid == -1:
        print(f"docno {docno} not found")
        continue
    index_docno = meta.getItem("docno" , docid)
    assert docno == index_docno

    document = meta.getItem("text" , docid)
    title = meta.getItem("title" , docid)

    topic = queries[queries['qid'] == qid]['query'].values[0]

    data_llm_dict = {'topic' : topic , 'document' : document , 'title' : title, 'qid' : qid , 'docno' : docno}
    data_llm_dict = query_llm(data_llm_dict)

    evaluated_df.loc[i , data_llm_dict.keys()] = data_llm_dict.values()
    with open(storage_data_name, "wb") as pickle_file:
        pickle.dump(evaluated_df, pickle_file)

# calculate the underlying correlation of machine annotation and human annotation
evaluated_df = evaluated_df.fillna(-2)

evaluated_df['label'] = evaluated_df['label'].astype(float)
evaluated_df['relevance_llm'] = evaluated_df['relevance_llm'].astype(float)
evaluated_df['quality_llm'] = evaluated_df['quality_llm'].astype(float)
evaluated_df['relevance_claude'] = evaluated_df['relevance_claude'].astype(float)
evaluated_df['relevance_gpt'] = evaluated_df['relevance_gpt'].astype(float)
evaluated_df['relevance_gemini'] = evaluated_df['relevance_gemini'].astype(float)


cut_off = len(evaluated_df['relevance_llm'])

# Calculate correlation coefficient
correlation = (evaluated_df['label'][:cut_off]).corr(evaluated_df['relevance_llm'][:cut_off])
print("Correlation coefficient:", correlation)
correlation = evaluated_df['label'][:cut_off].corr(evaluated_df['relevance_claude'][:cut_off])
print(f"Correlation for relevance claude: {correlation}")
correlation = evaluated_df['label'][:cut_off].corr(evaluated_df['relevance_gpt'][:cut_off])
print(f"Correlation for relevance gpt: {correlation}")
correlation = evaluated_df['label'][:cut_off].corr(evaluated_df['relevance_gemini'][:cut_off])
print(f"Correlation for relevance gemini: {correlation}")