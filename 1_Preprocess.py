# -*- coding: utf-8 -*-qu
"""
Basic main code. It does the first creation of the dataset we will
throughout this project.
--> Take the country, year and text of each constitution and
        preprocess the textual dataset.
@author: Marcos
"""
import glob
import os
import io
import re
import logging
import warnings
import pandas as pd

import nltk
from nltk.corpus import stopwords

import spacy
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS


def new_func():
    """Make it easier to see what is happening in LDA."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)


new_func()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Local of texts
# loc_en = r'data\eng'
# loc_es = r'data\es'
loc_total = r'data\simple_all'

# Getting the list of all texts
# en_texts = glob.glob(os.path.join(loc_en, '*.txt'))
# es_texts = glob.glob(os.path.join(loc_es, '*.txt'))
all_texts = glob.glob(os.path.join(loc_total, '*.txt'))


def get_df(texts):
    """
    Create a Df from a list of texts - their name in local memory.

    It makes a DF of the name, year (from its names) and content of each text.

    Parameters
    ----------
    texts : TYPE list
        DESCRIPTION. list of all txt files (constitutions) of the same
        language.


    Returns
    -------
    None.

    """
    country = []
    year = []
    constitution = []

    for txt in texts:
        country_year = os.path.splitext((os.path.basename(txt)))[0]
        country.append(' '.join(country_year.split('_')[:-1]))
        year.append(country_year.split('_')[-1])

        with io.open(txt, 'r', encoding='utf-8', errors='ignore') as fout:
            constitution.append(''.join(fout.readlines()))

    corpus = pd.DataFrame({'country': country, 'year': year,
                           'constitution': constitution})

    return corpus


def lemmatization(texts):
    """See https://spacy.io/api/annotation."""
    texts_out = []
    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(sent)
        texts_out.append([token.lemma_ for token in doc])
    return texts_out


# Creating DF and then applying lemmatization. Important to do before
# preprocessing.

# corpus_eng = get_df(en_texts)
# corpus_es = get_df(es_texts)
corpus_all = get_df(all_texts)

# data_eng = corpus_eng.constitution.to_list()
# eng_lemma = lemmatization(data_eng)
# eng_lemma = [' '.join(item) for item in eng_lemma]


# data_es = corpus_es.constitution.tolist()
# es_lemma = lemmatization(data_es)
# es_lemma = [' '.join(item) for item in es_lemma]

data_all = corpus_all.constitution.tolist()
all_lemma = lemmatization(data_all)
all_lemma = [' '.join(item) for item in all_lemma]

# Adding the new lemmatized data to the respective DFs
# corpus_eng['constitution_lemma'] = eng_lemma
# corpus_es['constitution_lemma'] = es_lemma
corpus_all['constitution_lemma'] = all_lemma

corpus_all.to_pickle(r'results/df_lemma(not_cleaned).pkl')


def clean_text(text):
    """
    Clean up text data.

    Make text lowercase, remove text in square brackets, remove
    punctuation, remove digits in general, remove urls, remove
    emails, remove "" characters and others.
    Also remove some common and bad words such as "Chapter" and "Copyright".
    """
    text = text.lower()
    text = re.sub(r"\(.*?\)", "()", text)
    text = re.sub(r"\(.*?\)", "<>", text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[!,:\-;\.\?\(\)]', '', text)
    text = ''.join(i for i in text if not i.isdigit())
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[‘’“”…]', '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub(r'<list>', ' ', text)
    text = re.sub(r'<title>', ' ', text)
    text = re.sub(r'<title', ' ', text)
    text = re.sub(r'<preamble>', ' ', text)
    text = re.sub(r'</list>', '', text)
    text = re.sub(r'/', '', text)
    text = re.sub(r'Chapter', '', text)
    text = re.sub(r'chapter', '', text)
    text = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', text)
    text = re.sub(r'section', '', text)
    text = re.sub(r'&', '', text)
    text = re.sub(r'-(?!\w)|(?<!\w)-', '', text)
    text = re.sub(r'_', '', text)
    text = re.sub(r'©', '', text)
    text = re.sub(r'@', '', text)

    return text


# Applying the cleaning approach to texts
# corpus_eng.constitution_lemma = corpus_eng['constitution_lemma'].apply(
#     lambda x: clean_text(x))
# corpus_eng.constitution = corpus_eng['constitution'].apply(
#     lambda x: clean_text(x))

# corpus_es.constitution_lemma = corpus_es['constitution_lemma'].apply(
#     lambda x: clean_text(x))
# corpus_es.constitution = corpus_es['constitution'].apply(
#     lambda x: clean_text(x))

corpus_all.constitution_lemma = corpus_all['constitution_lemma'].apply(
    clean_text)
corpus_all.constitution = corpus_all['constitution'].apply(
    clean_text)


# Add a column with the name of the language of each constitution
# corpus_eng['lang'] = 'eng'
# corpus_es['lang'] = 'es'

# Choose the best definition according to your needs!

# df_total = pd.concat([corpus_eng, corpus_es], ignore_index=True)
df_total = corpus_all

df_total.to_pickle('./results/df_total_all_set.pkl')
df_total.to_csv('./results/df_total_texts.csv')
# df_total = pd.read_csv('./results/df_total_texts.csv').iloc[:, 1:]
data = df_total.constitution_lemma.to_list()

# =============================================================================
# We create the list of stopwords
# =============================================================================
gensim_stop = STOPWORDS
nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'i', 'ii', 'iii',
                   'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii',
                   'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx',
                   'xxi', 'xxii', 'xxiii', 'xxiv', 'xxv', 'xxvi', 'xxvii',
                   'xxviii', 'xxix', 'xxx', 'xxxi', 'xxxii', 'xxxiii',
                   'xxxiv', 'xxxv', 'xxxvi', 'xxxvii', 'xxxviii', 'xxxix',
                   'xl', 'xli', 'xlii', 'xlii', 'xliii',  'xliv', 'xlv',
                   'xlvi', 'xlvii', 'xlviii', 'xlix', 'l', 'li', 'lii', 'liii',
                   'liv', 'lv', 'lvi', 'lvii', 'lviii', 'lix', 'lx', 'lxi',
                   'lxii', 'lxiii', 'lxiv', 'lxv', 'lxvi', 'lxvii', 'lxviii',
                   'lxix', 'lxx', 'lxxi', 'lxxii', 'lxxiii', 'lxxiv', 'lxxv',
                   'lxxvi', 'lxxvii', 'lxxviii', 'lxxix', 'lxxx', 'lxxxi',
                   'lxxxii', 'lxxxiii', 'lxxxiv', 'lxxxv',  'lxxxvi',
                   'lxxxvii', 'lxxxviii', 'lxxxix', 'xc', 'xci', 'xcii',
                   'xciii', 'xciv', 'xcv', 'xcvi', 'xcvii', 'xcviii', 'xcix',
                   'c', 'january', 'february', 'march', 'april', 'may',
                   'june', 'july', 'august', 'september', 'october',
                   'november', 'december', '+', 'shall', '*', 'copyright',
                   'project', 'constitutions', 'projects', 'stanford',
                   'havard', 'united nations', 'volume', 'ohio', 'preamble'
                   '©', 'bibliography', '<', '>', '/', 'may',
                   'constitution', 'country', 'shall', 'hoover', 'project',
                   'institution', 'new york', 'washington'])
stop_words.extend(gensim_stop)
stop_words.extend(['sub', 'clause', 'article'])


# =============================================================================
# Tokenizing options
# =============================================================================
def sent_to_words(sentences):
    """Do a simple tokenization."""
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def spacy_tokenize(sentences):
    """Do a simple tokenization with Spacy.

    Also removes stopwords and removes words that are not NOUN, VERB or ADV.
    """
    texts_out = []
    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    for text in sentences:
        doc = nlp(text)
        texts_out.append([token.text for token in doc if not token.is_stop
                          and token.pos_ in ['NOUN', 'VERB', 'ADJ']])

    text_final = [' '.join(item) for item in texts_out]

    return text_final


# =============================================================================
# Creating final objects
# =============================================================================
data_words = spacy_tokenize(data)
data_words = list(sent_to_words(data))


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    """Remove stopwords."""
    return [[word for word in simple_preprocess(str(doc)) if word not in
             stop_words] for doc in texts]


def make_bigrams(texts):
    """Create bigrams."""
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    """Make trigrams."""
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# Applying the functions just defined
data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostops)

# Save final results into a dataframe containing name and year of constitution
constitutions = df_total.copy()
constitutions['document'] = data_words_bigrams
constitutions.to_csv('results/csv/constitutions.csv')
