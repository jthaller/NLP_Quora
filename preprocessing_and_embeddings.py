import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc='Progress')

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
news_path = 'C:\\Users\\jerem\\OneDrive\\Documents\\Python\\NLP_Quora\\embeddings\\GoogleNews-vectors-negative300\\GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)

#Now we compare the overlap with these embeddings and the vocabulary we need for our quora questions
# It will output a list of out of vocabulary (oov) words that we can use to improve our preprocessing
import operator
def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:
            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x
oov = check_coverage(vocab,embeddings_index)

#these are the top words that are in our vocab but not the embedding
oov[:10]


for i in range(10):
    print(embeddings_index.index2entity[i])

# On first place there is "to". Why? Simply because "to" was removed when the GoogleNews
# Embeddings were trained. We will fix this later, for now we take care about the splitting
# of punctuation as this also seems to be a Problem. But what do we do with the punctuation
# then - Do we want to delete or consider as a token? I would say:
# It depends. If the token has an embedding, keep it, if it doesn't we don't need it anymore. So lets check:
'?' in embeddings_index
'&' in embeddings_index

def clean_text(x):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x

train["question_text"] = train["question_text"].progress_apply(lambda x: clean_text(x))
sentences = train["question_text"].apply(lambda x: x.split())
vocab = build_vocab(sentences)

oov = check_coverage(vocab,embeddings_index)
oov[:10]


# hmm why is "##" in there? Simply because as a reprocessing all numbers bigger tha 9 have been replaced by hashs.
# I.e. 15 becomes ## while 123 becomes ### or 15.80€ becomes ##.##€. So lets mimic this preprocessing step to
# further improve our embeddings coverage
import re
def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

train["question_text"] = train["question_text"].progress_apply(lambda x: clean_numbers(x))
sentences = train["question_text"].progress_apply(lambda x: x.split())
vocab = build_vocab(sentences)

oov = check_coverage(vocab,embeddings_index)
oov[:20]



# take care of mispellings, british -> us english, social media stuff and modern colloquialisms,
# simply remove the words "a","to","and" and "of" since those have obviously been downsampled
# when training the GoogleNews Embeddings.
# multi regex script to do the replacing
def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium'

                }
mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)


train["question_text"] = train["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
sentences = train["question_text"].progress_apply(lambda x: x.split())
to_remove = ['a','to','of','and']
sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]
vocab = build_vocab(sentences)

oov = check_coverage(vocab,embeddings_index)
oov[:20]






# from textacy import preprocess
#
# text = preprocess.preprocesstext(text, fixunicode=True,
# lowercase=False,
# transliterate=True,
# nourls=True, noemails=True,
# nophonenumbers=True,
# nonumbers=True, nocurrencysymbols=True, nopunct=False,
# nocontractions=True, noaccents=True)
#
# text = preprocess.normalize_whitespace(text)
