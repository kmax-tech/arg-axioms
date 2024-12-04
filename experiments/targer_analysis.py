import pyterrier as pt
import pickle
import utils.repair_result_dataframe as rd
if not pt.java.started() :
    pt.java.init()
import settings as s
from utils.send_data_to_socket import send_data_to_socket
import numpy as np
if __name__ == "__main__":

    experiment_name = "targer_analysis_touche21"
    experiment_save_path = s.PROJECT_ROOT / s.SAVE_PATH / experiment_name


    dataset = pt.datasets.get_dataset(f'irds:{s.dataset}')

    topics = dataset.get_topics()
    qrels = dataset.get_qrels()
    topic_qrels = dataset.get_topicsqrels()
    indexref = pt.IndexRef.of(str(s.dataset_index_dir))

    dirichletLM = pt.terrier.Retriever(str(s.dataset_index_dir), wmodel="DirichletLM")

    res_df = dirichletLM.transform(topics)

    filtered_df = res_df[(res_df['rank'] >= 0) & (res_df['rank'] <= 20)]

    result = filtered_df['docno'].to_list()
    resultids = filtered_df['docid'].to_list()

    index = pt.IndexFactory.of(indexref)
    meta = index.getMetaIndex()

    filtered_documents = {
    s.docnos : [meta.getItem("docno",x) for x in resultids],
    s.texts : [meta.getItem("text",x) for x in resultids],
    s.TASK : s.TARGER_ANALYSIS,
    s.SOCKET_DATASET_KEY : s.dataset_short
    }

    assert len(filtered_documents[s.texts]) == len(filtered_documents[s.docnos])
    if experiment_save_path.exists():
        with open(experiment_save_path, 'a') as f:
            received_data = pickle.load(f)
    else:
        received_data = send_data_to_socket(filtered_documents)

    def analyze_sentence_length(document):
        sentence_length = [len(sentence) for sentence in document]
        mean = np.mean(sentence_length)
        return mean


    document_sentences = received_data[s.sentences]
    document_arguments = received_data[s.argument_units]

    average_sentence_length = analyze_sentence_length(document_sentences)
    average_argument_length = analyze_sentence_length(document_arguments)

    average_sentence_document = np.mean(average_sentence_length)
    average_argument_document = np.mean(average_argument_length)

    print(f"Average sentence document length: {average_sentence_document}")
    print(f"Average argument document length: {average_argument_document}")









