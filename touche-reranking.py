import json

from pathlib import Path
import sim_calc as sc
import settings as s
import pickle
import sys
import random
import utils as u
import json
import pyterrier as pt
import pandas  as pd

random.seed(42)

# extract corresponding spacy terms

RESULTS_NAME = "results.jsonl"
INDEX_LOCATION = "pyterrier_index"

class BASEJSONL():
    def create_jsonl(self, prefix):
        new_name = Path.cwd().joinpath(prefix, RESULTS_NAME)
        dir = new_name.parent
        dir.mkdir(parents=True, exist_ok=True)
        with open(new_name, "w") as f:
            for d in self.data_for_submission:
                json.dump(d, f)
                f.write('\n')

class Pyterrier(BASEJSONL):

    def __init__(self):
        super().__init__()
        if not pt.started():
            pt.init()
        self.index_location = Path.cwd() / INDEX_LOCATION

        self.index_location.mkdir(parents=True,exist_ok=True)

        self.data_for_submission = []

        self.arguments_id_dict = u.ParseArgusXML().arguments_id_dict
        self.Index = pt.IterDictIndexer(str(self.index_location),meta={'docno': 20, 'image': 100})

    def iter_data(self):
        files_sorted = sorted([x for x in u.DIRECTORY_WITH_RETRIEVED_IMAGES.iterdir()])
        docno = 1
        for dir in files_sorted:
            dir_sorted = sorted([x for x in dir.iterdir()])
            for arg in dir_sorted:
                id = str(arg.stem).strip()
                caption_path = arg
                with open(caption_path,"r") as f:
                    text = f.read().strip()
                if len(text) == 0:
                    continue
                yield {'docno': str(docno), 'text': text, 'image': id}
                docno += 1

    def index_files(self):
        self.Index.index(self.iter_data())

    def meta_query(self):

        # results = bm25.search(query)

        doc = self.Index.get_document(7746)

    def data_exp(self):
        self.meta_index = pt.IndexFactory.of(
            "/home/max/PycharmProjects/imageDescription/baseline/pyterrier_index/data.properties")

        met = self.meta_index.getMetaIndex()


        queries = self.create_queries()
        bm25 = pt.BatchRetrieve(self.Index, wmodel="BM25")
        res_df = bm25.transform(queries)
        unique_argu_ids = res_df['qid'].unique()

        for x in unique_argu_ids:
            argu_ranks = res_df[res_df['qid'] == x]

            for rank in range(0,11):
                image_id_df = argu_ranks[argu_ranks["rank"]==rank]
                assert image_id_df.shape[0] == 1
                docno = image_id_df.iloc[0]["docno"]
                docid = image_id_df.iloc[0]["docid"]

                ind_docno =  met.getItem("docno", docid)
                assert docno == ind_docno

                image_id = met.getItem("image", docid)

                data_obj = {
                    s.ARGU_ID: x,
                    s.METHOD: s.RETRIEVAL,
                    s.IMAGE_ID: image_id,
                    s.RATIONALE: "",
                    s.RANK: rank + 1,
                    s.TAG: "touche24 image retrieval - bm25 baseline"
                }

                self.data_for_submission.append(data_obj)

    def create_queries(self):

        qids = []
        queries = []
        for argu_id, arg in self.arguments_id_dict.items():
            qids.append(argu_id)

            prem = arg[s.PREMISE]
            prem_cleaned = prem.replace("'","").strip()
            prem_cleaned = prem_cleaned.replace("?","").strip()

            queries.append(prem_cleaned)
        final_dict = {"qid": qids, "query" : queries}

        df = pd.DataFrame.from_dict(final_dict)
        return df


class Retrieval(BASEJSONL):

    def __init__(self):
        super().__init__()
        self.nbr_embeds = 10
        self.arguments_id_dict = u.ParseArgusXML().arguments_id_dict

        if not u.IMAGE_EMBEDDINGS_PATH.exists():
            print("Could not find Image Embeddings")
            sys.exit(1)
        if not u.ARGU_EMBEDDINGS_PATH.exists():
            print("Could not find Argu Embeddings")
            sys.exit(1)

        with open(u.IMAGE_EMBEDDINGS_PATH, 'rb') as f:
            self.image_embedding_dict = pickle.load(f)

        with open(u.ARGU_EMBEDDINGS_PATH, 'rb') as f:
            self.argu_embedding_dict = pickle.load(f)

        self.data_for_submission = []

    def create_sim_images(self):
        for argu_id, argu_data in self.argu_embedding_dict.items():
            argu_embeddings = argu_data["embeddings"]  # get the calculated embeddings
            top_n_embeddings = []

            for image_id, image_data in self.image_embedding_dict.items():

                image_data_embeddings = image_data["embeddings"]
                if len(image_data_embeddings) == 0:
                    continue
                similarity = sc.get_mean_sim(argu_embeddings, image_data_embeddings)
                top_n_embeddings.append((image_id, similarity))
                if len(top_n_embeddings) <= self.nbr_embeds:
                    continue
                else:
                    embeddings_sorted = sorted(top_n_embeddings, key=lambda x: x[1], reverse=True)
                    top_n_embeddings = embeddings_sorted[:self.nbr_embeds]

            for ind, retrieved_id_data in enumerate(top_n_embeddings, 1):
                rationale = self.image_embedding_dict[retrieved_id_data[0]]["text"]

                data_obj = {
                    s.ARGU_ID: argu_id,
                    s.METHOD: s.RETRIEVAL,
                    s.IMAGE_ID: retrieved_id_data[0],
                    s.RATIONALE: rationale,
                    s.RANK: ind,
                    s.TAG: "touche24 image retrieval - sbert mean baseline"
                }

                self.data_for_submission.append(data_obj)

    def create_random_baseline(self):
        for argu_id, argu_data in self.arguments_id_dict.items():

            # get 5 random keys
            top_five_embeddings = random.sample(self.image_embedding_dict.keys(), 5)

            for ind, retrieved_id_data in enumerate(top_five_embeddings, 1):
                data_caption_path = self.image_embedding_dict[retrieved_id_data][2]
                with open(data_caption_path, "r") as f:
                    rationale = f.read()

                data_obj = {
                    s.ARGU_ID: argu_id,
                    s.METHOD: s.RETRIEVAL,
                    s.IMAGE_ID: retrieved_id_data,
                    s.RATIONALE: rationale,
                    s.RANK: ind,
                    s.TAG: "touche24 image retrieval - random baseline"
                }

                self.data_for_submission.append(data_obj)




class Generation(BASEJSONL):
    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix
        self.prefix_path = Path.cwd().joinpath(prefix)
        self.prefix_image_path = Path.cwd().joinpath(prefix, s.IMAGE_DIR)
        self.arguments_id_dict = u.ParseArgusXML().arguments_id_dict

        self.prefix_image_path.mkdir(parents=True, exist_ok=True)

        self.data_for_submission = []

    def create_images_generation(self):
        cnt = 1
        for argu_id, argu_data in self.arguments_id_dict.items():
            print(f"Generating argument {cnt}")
            cnt += 1
            premise = argu_data["premise"]

            #gi.generate_images(premise, argu_id, 5, self.prefix_image_path)

            for i in range(1,6):

                data_obj = {
                    s.ARGU_ID: argu_id,
                    s.METHOD: s.GENERATION,
                    s.PROMPT: premise,
                    s.IMAGE_NAME : f"{argu_id}-{i}.jpg",
                    s.RATIONALE: premise,
                    s.RANK: i,
                    s.TAG: "touche24 image retrieval - random baseline"
                }

                self.data_for_submission.append(data_obj)



if __name__ == "__main__":
    #u.PremiseArgumentEmbedding()
    #u.ImageEmbedding()
    x = Retrieval()
    x.create_sim_images()
    x.create_jsonl("sbert-embed")
    # x.create_jsonl("sim_style")
    #x.create_random_baseline()
    #x.create_jsonl("random")
    #x = Generation("generation")
    #x.create_images_generation()
    # x.create_jsonl()
    pyt = Pyterrier()
    #pyt.index_files()

    pyt.data_exp()
    pyt.create_jsonl("bm25")
