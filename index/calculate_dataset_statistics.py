from pyterrier import init, started, set_property
if not started():
    init()
set_property("metaindex.compressed.reverse.allow.duplicates", "true")
from pyterrier.datasets import get_dataset
from pyterrier.batchretrieve import BatchRetrieve
from pathlib import Path
from ir_axioms.backend.pyterrier.experiment import AxiomaticExperiment
from ir_measures import nDCG, Bpref
from pyterrier.index import IterDictIndexer
import pandas as pd
from tabulate import tabulate
import string
from nltk.tokenize import sent_tokenize


dataset_name = "argsme/2020-04-01/touche-2021-task-1"
dataset = get_dataset(f"irds:{dataset_name}")


docnos = dataset.get_qrels()["docno"].values


def generate_filter_qrels():
    for d in dataset.get_corpus_iter():
        if (d["docno"] in docnos):
            yield d

def calculate_values(data):

    print(len(data))
    data = pd.DataFrame(data)

    quantile_interpolation = "linear"

    data_mean = data.mean()
    data_05 = data.quantile(0.05)
    data_25 = data.quantile(0.25)
    data_median = data.quantile(0.5)
    data_75 = data.quantile(0.75)
    data_95 = data.quantile(0.95)
    data_max = data.max()
    data_min = data.min()
    data_std = data.std()

    # names for included columns
    cols_dict = [
            ("Mean", data_mean[0]),
            ("05 Perc.", data_05[0]),
            ("25 Perc.", data_25[0]),
            ("Median", data_median[0]),
            ("75 Perc.", data_75[0]),
            ("95 Perc.", data_95[0]),
            ("Max", data_max[0]),
            ("Min", data_min[0]),
            ("Std.", data_std[0]) ]
    return cols_dict



if __name__ == "__main__":

    calculate_values([1,2,3])

    total_length = []

    for x in generate_filter_qrels():
        premise = x["premises_texts"]
        translator = str.maketrans("", "", string.punctuation)
        clean_sentence = premise.lower().translate(translator)
        words = clean_sentence.split()
        words_nbr = len(words)

        total_length.append(words_nbr)

    res = calculate_values(total_length)
    x = tabulate(res,tablefmt="latex_raw", headers=["Metric", "Nbr. of Words"])
    print(x)

"""
dirichletLM = BatchRetrieve(str(index_dir.absolute()), wmodel="DirichletLM")

experiment = AxiomaticExperiment(
    retrieval_systems=[dirichletLM],
    topics=dataset.get_topics(),
    qrels=dataset.get_qrels(),
    index=index_dir,
    axioms=[],
    cache_dir=cache_dir,
    depth=5,
    filter_by_qrels=False,
)
with open(Path(__file__).parent / "test.txt", "w") as t:
    t.write(experiment.preferences.to_string())
"""