
import llm_finetune.prompting_template as pt
import settings as s
import json
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split


def load_data(path):
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

touche20_path = s.PROJECT_ROOT / '_data_for_llm_finetuning' / 'touche20_data.jsonl'
touche21_path = s.PROJECT_ROOT / '_data_for_llm_finetuning' / 'touche21_data.jsonl'

touche20_data = load_data(touche20_path)
touche_21_data = load_data(touche21_path)

random.seed(42)
random.shuffle(touche20_data)

touche20_train_data, touche20_temp_data = train_test_split(touche20_data, test_size=0.3, random_state=42)
touche20_test_data, touche20_dev_data = train_test_split(touche20_temp_data, test_size=1 / 3, random_state=42)

random.seed(42)

random.shuffle(touche20_data)
touche21_train_data, touche21_temp_data = train_test_split(touche_21_data, test_size=0.3, random_state=42)
touche21_test_data, touche21_dev_data = train_test_split(touche21_temp_data, test_size=1 / 3, random_state=42)

data_train_combined =touche20_train_data + touche21_train_data
data_dev_combined = touche20_dev_data + touche21_dev_data

def create_touche_dataset(data_list,name,sample_size=None):

    data_dict_to_parse = defaultdict(list)
    for entry in data_list :
        data_dict_to_parse[entry['label']].append(entry)

    final_data = []
    for label,documents in data_dict_to_parse.items():
        sample = documents
        if not sample_size is None:
            if len(documents) > sample_size:
                sample = random.sample(documents,sample_size)
        sample_formatted = pt.create_prompt_list(sample)
        final_data += sample_formatted

    if sample_size is None :
        sample_size = "all"

    file_name = f"name_{name}_sample_size_{sample_size}.jsonl"

    file_name_path = s.PROJECT_ROOT / '_data_for_llm_finetuning' / file_name

    with open(file_name_path, "w") as f :
        for item in final_data :
            f.write(json.dumps(item) + "\n")

    # also send data directly to storage in GPU CLUSTER

if __name__ == "__main__":

    create_touche_dataset(data_train_combined,"touche_all_train_finetune",sample_size=400)
    create_touche_dataset(data_dev_combined,"touche_all_dev_finetune",sample_size=None)
    create_touche_dataset(touche20_test_data,"touche20_test_finetune",sample_size=None)
    create_touche_dataset(touche21_test_data,"touche21_test_finetune",sample_size=None)