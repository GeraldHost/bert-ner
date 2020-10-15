import transformers
import os

dirname = os.path.dirname(__file__)

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = "bert-base-uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = os.path.join(dirname, 'input/input.csv')
EVAL_FILE = os.path.join(dirname, 'input/input.csv')
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True
)
