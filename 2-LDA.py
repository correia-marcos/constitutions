# -*- coding: utf-8 -*-
"""
Basic main code.

@author: marcola
"""
import glob
import os
import pandas as pd
import io
import re
from pprint import pprint
from collections import Counter

import nltk
from nltk.corpus import stopwords

import spacy
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt

import logging
import warnings

# Not important to understand, make it easier to see what is happenning in LDA
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
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


# Adding the new lemmatized data to the respectves DFs
# corpus_eng['constitution_lemma'] = eng_lemma
# corpus_es['constitution_lemma'] = es_lemma
corpus_all['constitution_lemma'] = all_lemma

corpus_all.to_pickle(r'results/df_lemma(not_cleaned).pkl')


def clean_text(text):
    """
    Clean up text data.

    Make text lowercase, remove text in square brackets, remove
    punctuation, remove digits in general, remove urls, remove
    emails and remove "" caracteres.
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


# Apllying the cleaning approach to texts
# corpus_eng.constitution_lemma = corpus_eng['constitution_lemma'].apply(
#     lambda x: clean_text(x))
# corpus_eng.constitution = corpus_eng['constitution'].apply(
#     lambda x: clean_text(x))

# corpus_es.constitution_lemma = corpus_es['constitution_lemma'].apply(
#     lambda x: clean_text(x))
# corpus_es.constitution = corpus_es['constitution'].apply(
#     lambda x: clean_text(x))

corpus_all.constitution_lemma = corpus_all['constitution_lemma'].apply(
    lambda x: clean_text(x))
corpus_all.constitution = corpus_all['constitution'].apply(
    lambda x: clean_text(x))


# Add a column with the name of the language of each constitution
# corpus_eng['lang'] = 'eng'
# corpus_es['lang'] = 'es'

# Choose the best definition according to your needs!

# df_total = pd.concat([corpus_eng, corpus_es], ignore_index=True)
df_total = corpus_all

df_total.to_pickle('./results/df_total_all_set.pkl')
df_total.to_csv('./results/df_total_texts.csv')
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


# Applyinh the functions just defined
data_words_nostops = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostops)

# Save final results into a dataframe containing name and year of constitution
constitutions = df_total.copy()
constitutions['document'] = data_words_bigrams
constitutions.to_csv('results/csv/constitutions.csv')

# Create Dictionary
id2word = corpora.Dictionary(data_words_bigrams)
# id2word.filter_extremes()

# Create Corpus
texts = data_words_bigrams

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Get the number of differents words in the corpus
n_words = id2word.num_pos
# Get the most common 50 words in the corpus (possiblity removing them as
# stopwords in further runs)
words_list = [word for words in data_words for word in words]
most_common_words = Counter(words_list).most_common(50)

# =============================================================================
# Measuring best number of topics
# =============================================================================


def compute_coherence_values(dictionary, corpus, texts, limit, start=2,
                             step=3):
    """
    Compute c_v coherence for various number of topics.

    Parameters
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with
    respective number of topics.
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics,
                                                random_state=42,
                                                iterations=200,
                                                update_every=1,
                                                passes=25,
                                                alpha='auto',
                                                eta='auto',
                                                per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts,
                                        dictionary=dictionary,
                                        coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


model_list, coherence_values = compute_coherence_values(dictionary=id2word,
                                                        corpus=corpus,
                                                        texts=texts,
                                                        start=15,
                                                        limit=100,
                                                        step=5)

# Plot the results!
start, limit, step = 15, 100, 5
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# See the results in a different way
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# Get the best model or the prefered one
def get_model(list_coherence, list_models, best_model=True, non_optimal=50):
    """
    Return the model selected in a list of models.

    Parameters
    ----------
    list_coherence : List
        DESCRIPTION. List containing all coherence values to be compared
    list_models : List
        DESCRIPTION. List with all model in the same order of list_coherence
    best_model : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    Optimal model and the number of topics

    """
    range_topics = range(15, 100, 5)
    index_non_optimal = list(range_topics).index(non_optimal)
    index_max = max(range(len(list_coherence)),
                    key=list_coherence.__getitem__)
    # Get the number of topics in the best model or the one choosen
    if best_model:
        optimal_model = list_models[index_max]
        topics_number = range_topics[index_max]
    else:
        optimal_model = list_models[index_non_optimal]
        topics_number = range_topics[index_non_optimal]

    return optimal_model, topics_number


optimal_model, topics_number = get_model(coherence_values, model_list,
                                         best_model=False, non_optimal=50)

# Create list with Topics words and save


def get_listTopics(topics, lda_model, dictionary):
    """Create a list of 15 most import words for each topic."""
    topic_words = []
    for i in range(topics):
        tt = lda_model.get_topic_terms(i, 15)
        topic_words.append([dictionary[pair[0]] for pair in tt])

    return topic_words


words = get_listTopics(topics_number, optimal_model, id2word)
words_df = pd.DataFrame(words)
words_df.to_csv('results/Topics_models/words_50topics(08_05).csv')

# =============================================================================
# Specific model - not necessary to run
# =============================================================================

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=15,
                                            random_state=42,
                                            iterations=200,
                                            update_every=1,
                                            passes=50,
                                            alpha='auto',
                                            eta='auto',
                                            per_word_topics=True)

optimal_model = lda_model
doc_lda = optimal_model[corpus]
pprint(lda_model.print_topics())

# lda_model.save('results/lda_model/lda_15_topics')
# id2word.save('results/lda_model/ldaDict_15')

print('\nPerplexity: ', lda_model.log_perplexity(corpus))

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=texts,
                                     dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# =============================================================================
# Final results with optimal_model
# =============================================================================

# Create dataset with topics and constitutions
optimal_model.get_document_topics(corpus, minimum_probability=0.0)
docTopics = []
for i, row in enumerate(optimal_model[corpus]):
    row = sorted(row, key=lambda x: (x[0]), reverse=False)
    docTopics.append(row)

topicSeriesDf = pd.DataFrame([[y[1] for y in x] for x in docTopics])

df_final = df_total[['country', 'year']]
df_final = df_final.join(topicSeriesDf)

# Note that you must change the number of columns if the optimal topics
# is different. This should have be done as an automatized function, but
# I was a little low in time.

df_final.columns = ['country', 'year', 'Topic 1', 'Topic 2', 'Topic 3',
                    'Topic 4', 'Topic 5', 'Topic 6', 'Topic 7', 'Topic 8',
                    'Topic 9', 'Topic 10', 'Topic 11', 'Topic 12', 'Topic 13',
                    'Topic 14', 'Topic 15', 'Topic 16', 'Topic 17', 'Topic 18',
                    'Topic 19', 'Topic 20', 'Topic 21', 'Topic 22', 'Topic 23',
                    'Topic 24', 'Topic 25', 'Topic 26', 'Topic 27', 'Topic 28',
                    'Topic 29', 'Topic 30', 'Topic 31', 'Topic 32', 'Topic 33',
                    'Topic 34', 'Topic 35', 'Topic 36', 'Topic 37', 'Topic 38',
                    'Topic 39', 'Topic 40', 'Topic 41', 'Topic 42', 'Topic 43',
                    'Topic 44', 'Topic 45', 'Topic 46', 'Topic 47', 'Topic 48',
                    'Topic 49', 'Topic 50']

folder = r'results/Topics_models/'
df_final.to_pickle(rf'{folder}/df_topics(50topics_08_05(full_set)).pkl')


vis = gensimvis.prepare(optimal_model, corpus, id2word)
pyLDAvis.enable_notebook()
pyLDAvis.display(vis)
pyLDAvis.save_html(vis, f'{folder}/lda_50topics_all_set(08_05).html')
