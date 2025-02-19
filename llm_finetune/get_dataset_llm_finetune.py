# get data needed from dataset, in order to finetune a llama llm
import pyterrier as pt
from pyterrier.datasets import get_dataset
import utils.get_datafeatures_from_datasets as gdf
if not pt.java.started() :
    pt.java.init()
import settings as s
import json
from collections import defaultdict
def get_data_for_llm_finetune(dataset_name,topic_key):
    print(f"Getting data for {dataset_name}")

    # get all human annotated qrels
    s.set_data_manually(dataset_name)

    dataset = get_dataset(f'irds:{s.dataset}')
    data_qrels = dataset.get_qrels()
    queries = dataset.get_topics()

    indexref = pt.IndexRef.of(str(s.dataset_index_dir))
    index = pt.IndexFactory.of(indexref)
    meta = index.getMetaIndex()

    data_list_to_save = []
    for i,x in data_qrels.iterrows():
        a = x.to_dict() # convert entry to dict for further processing
        del a['iteration']

        docno = x['docno']
        qid = x['qid']
        docid = meta.getDocument("docno" , docno)
        if docid == -1:
            print(f"docno {docno} not found")
            continue
        document = meta.getItem("text", docid)
        title = meta.getItem("title", docid)
        topic = queries[queries['qid'] == qid][topic_key].values[0]
        a['topic'] = gdf.process_topic_entry(topic)
        a['title'] = title
        a['document'] = document
        a['dataset'] = dataset_name
        print(i)
        data_list_to_save.append(a)

    # File path for the JSONL file
    file_path = s.PROJECT_ROOT / '_data_for_llm_finetuning' / f'{dataset_name}_data.jsonl'
    # Open the file in write mode
    with open(file_path, 'w') as file:
        for item in data_list_to_save:
            # Convert the dictionary to a JSON string and write it to the file
            json_line = json.dumps(item)
            file.write(json_line + '\n')
    return data_list_to_save

if __name__ == '__main__':

    touche_20_data = get_data_for_llm_finetune('touche20',"text")
    touche_21_data = get_data_for_llm_finetune('touche21',"query")

