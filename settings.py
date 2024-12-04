import sys
from pathlib import Path
import configs as cfg
import os
from dotenv import load_dotenv
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT=Path('/Users/max/projects/axiomatic-reranking')

axioms_cache_embeddings = PROJECT_ROOT / '_axioms_cache_embeddings'
cache_dir = axioms_cache_embeddings

dotenv_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path)
dataset_name = os.environ.get("dataset")

# has to be set by each implementation
data = None
dataset = None
dataset_short = None
dataset_index_dir = None

def set_data_manually(dataset_name): # in case data is specified manually
    global dataset
    global dataset_short
    global dataset_index_dir
    global data
    if dataset_name not in cfg.configs:
        sys.exit(f"Dataset {dataset_name} not found in configs")
    data = cfg.configs[dataset_name]
    dataset = data['dataset']
    dataset_short = data['dataset_short']
    dataset_index_dir = PROJECT_ROOT / f'_{dataset_short}_index'




dotenv_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path)
key = os.environ.get("OPENAI-KEY")

CURRENT_EXPERIMENT_NAME = None

BASE_RETRIEVER = 'DirichletLM'
SBERT = 'sbert'
ADA = 'ada'
SMTC1 = 'stmc1'
#EMBDEDDINGS_STYLES = [ADA]

sentences = 'sentences'
argument_units = 'arguments'
SENTENCE_STYLES = [sentences , argument_units]

docnos = 'docnos'
texts = 'texts'
TASK = 'task'


# Socket Terms



DOCUMENT_RANKING = 'DOCUMENT_RANKING'
SOCKET_DOCUMENT1  = 'DOCUMENT1'
SOCKET_DOCUMENT2 = 'DOCUMENT2'
SOCKET_QUERY = 'QUERY'
SOCKET_COMPARE_METHOD = 'COMPARE_METHOD'
EMBEDD_SINGLE = 'EMBEDD_SINGLE'
BATCH_EMBEDDING = 'BATCH_EMBEDDING'
TARGER_ANALYSIS = 'TARGER_ANALYSIS'
MAX = 'MAX'
MEAN = 'MEAN'
SOCKET_DATASET_KEY = 'DATASET'

OK = 'OK'
request = 'request'
socket = 'socket'
server_communication_method = socket # specify communications style with Embeddings Getter

server_port = 8080
server_embedding_url = 'http://localhost:{}/get_embedding'.format(server_port)

# Terms for Saving
SAVE_PATH = '_experiments_results_raw'

# TERMS FOR TOUCHE
TOUCHE_DIR = None


# TERMS FOR LLM PROMPTING
OPEN_AI_MODEL = "gpt-4o-2024-08-06"
GEMINI_MODEL = "gemini-1.5-flash-002"
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
TOP_P = 0.1
TEMPERATURE = 0.2
GEMINI_CRED = '/Users/max/projects/argument-classification-walton-schemes/cred/argu-walton-class-gen-e8a22ff2d56d.json'

# TERMS FOR TARGER EMBEDDINGS, are stored in TASK_INFO dict
TASK_INFO_TARGER_OWN_SENTENCIZER= 'TARGER_OWN_SENTENCIZER'

# TERMS FOR IDENTIFYING
IDENT_AUS_SINGLE_SENTENCE = 'AUS_SINGLE_SENTENCE'
IDENT_AUS_FULL_DOCUMENT= 'AUS_FULL_DOCUMENT'
IDENT_SENTENCES = 'SENTENCES'
SOCKET_IDENTIFIER = 'SOCKET_IDENTIFIER'
SOCKET_TASK_INFO = 'SOCKET_TASK_INFO'
SOCKET_TASK = 'SOCKET_TASK'
SOCKET_SENTENIZE = 'SOCKET_SENTENIZE'
SOCKET_EMBEDDING_STYLE = 'SOCKET_EMBEDDING_STYLE'

SOCKET_DOCNO = 'SOCKET_DOCNO'
SOCKET_DOCUMENT = 'SOCKET_DOCUMENT'

# Settings for Evaluation of QRELS
ONLY_HUMAN_QRELS = True
QRELS_TO_USE = None

# Terms for specifying the type llms
CLAUDE = 'claude'
GPT = 'gpt'
GEMINI = 'gemini'
LLMS_TO_USE = ['claude']

# Terms for Multiprocessing
NBR_PROCESSES = 0

TARGER_TOTAL_TRIES = 3