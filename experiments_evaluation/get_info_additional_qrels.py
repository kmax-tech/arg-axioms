import settings as s
import pyterrier as pt
import pickle
import utils.repair_result_dataframe as rd

data_all_instances_to_annotate = []
if __name__ == "__main__":

    for dir in s.ADDITIONAL_QREL_LOCATION.iterdir():
        dataset = dir.name
        s.set_data_manually(dataset)
        queries = rd.get_dataset_queries()
        indexref = pt.IndexRef.of(str(s.dataset_index_dir))
        index = pt.IndexFactory.of(indexref)
        # inv = index.getInvertedIndex()
        meta = index.getMetaIndex()

        for file in dir.iterdir():
            experiment_name = file.name.replace(s.ADDITIONAL_QREL_FILE_ENDING, "")
            with open(file, 'rb') as f:
                df = pickle.load(f)
            for i, x in df.iterrows() :
                docid = x['docid']
                qid = x['qid']
                docno = x['docno']
                index_docno = meta.getItem("docno", docid)
                assert docno == index_docno

                document = meta.getItem("text", docid)
                title = meta.getItem("title", docid)

                topic = queries[queries['qid'] == qid]['query'].values[0]
                assert topic == x['query']
                data_dict = {'dataset' : dataset, 'topic' : topic, 'document' : document, 'title' : title,
                             'docno' : docno,
                             'qid' : qid}
                data_all_instances_to_annotate.append(data_dict)




