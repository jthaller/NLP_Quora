import time
import random
import pandas as pd
import numpy as np
import gc
import re
import torch
from torchtext import data
import spacy
from tqdm import tqdm_notebook, tnrange
from tqdm.auto import tqdm

tqdm.pandas(desc='Progress')
from collections import Counter
from textblob import TextBlob
from nltk import word_tokenize

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torchtext.data import Example
from sklearn.metrics import f1_score
import torchtext
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# cross validation and metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.optim.optimizer import Optimizer
from unidecode import unidecode

# defining a seed makes random numbers reproducible.
# example:
# random.seed(10)
# random.random()
# always returns 0.5714025946899135
def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()

SEED=12345
embed_size = 300 # how big is each word vector
max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 80 # max number of words in a question to use
batch_size = 256 # how many samples to process at once
n_epochs = 5 # how many times to iterate over all samples
n_splits = 5 # Number of K-fold Splits


df_train = pd.read_csv("C:\\Users\\jerem\\OneDrive\\Documents\\Python\\NLP_Quora\\train.csv")
df_test = pd.read_csv("C:\\Users\\jerem\\OneDrive\\Documents\\Python\\NLP_Quora\\test.csv")
df = pd.concat([df_train ,df_test],sort=True)

df_train.head()
df_test.head()
df_train.shape
df_test.shape
