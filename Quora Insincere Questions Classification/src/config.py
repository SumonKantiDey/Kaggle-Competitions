import transformers
from transformers import TFAutoModel, AutoTokenizer

MAX_LEN = 64
TRAIN_BATCH_SIZE = 192
VALID_BATCH_SIZE = 128
EPOCHS = 3
MODEL_SAVE_PATH = '../models/insincerity_model.pt'
TRAINING_FILE = "../input/train.csv"
TEST_FILE = "../input/test.csv"
TOKENIZER = AutoTokenizer.from_pretrained("xlm-roberta-base")