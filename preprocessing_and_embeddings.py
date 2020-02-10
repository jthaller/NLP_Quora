import pandas as pd
from tqdm import tqdm
tqdm.pandas()

train = pd.read_csv("C:\\Users\\jerem\\OneDrive\\Documents\\Python\\NLP_Quora\\train.csv")
test = pd.read_csv("C:\\Users\\jerem\\OneDrive\\Documents\\Python\\NLP_Quora\\test.csv")
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)

def build_vocab(sentences, verbose =  True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab


sentences = train["question_text"].progress_apply(lambda x: x.split()).values #split every sentence up into words z.b. [how, do, you, etc.]
vocab = build_vocab(sentences)
print({k: vocab[k] for k in list(vocab)[:5]})


#import pre-defined embeddings. These ones are from a GoogleNews articles
from gensim.models import KeyedVectors
news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)

#Now we compare the overlap with these embeddings and the vocabulary we need for our quora questions
# It will output a list of out of vocabulary (oov) words that we can use to improve our preprocessing
