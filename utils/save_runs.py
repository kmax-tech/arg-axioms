import settings as s
from slugify import slugify
import pickle
import pandas as pd



def save_runs(data,name=None):
    if name is None:
        name = s.CURRENT_EXPERIMENT_NAME

    path_to_save = s.PROJECT_ROOT / s.SAVE_PATH
    path_to_save.mkdir(parents=True, exist_ok=True)

    named_slug = slugify(name,allow_unicode=True)

    path_to_save_full = path_to_save / (named_slug + ".pkl")
    # if isinstance(data, pd.DataFrame) :
        # path_to_save_full = path_to_save / (named_slug + ".csv")


    with open(path_to_save_full, 'wb') as f:
       # if isinstance(data, pd.DataFrame):
           # data.to_csv(path_to_save_full, index=False, header=True)
        pickle.dump(data, f)

# load existing data
def load_runs(name):
    named_slug = slugify(name,allow_unicode=True)

    if not named_slug.endswith(".pkl"):
        named_slug = named_slug + ".pkl"
    path_to_load_full = s.PROJECT_ROOT / s.SAVE_PATH / named_slug
    if not path_to_load_full.exists():
        print(f"Loading data from {path_to_load_full} does not exist")
        return None
    with open(path_to_load_full, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    a = slugify('DirichletLM-Reranking')
    print(a)