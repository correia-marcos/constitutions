# -*- coding: utf-8 -*-
"""
Creates the LDA object.

@author: Marcos
"""
from pprint import pprint
from collections import Counter
import pandas as pd
import ast

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt

import logging
import warnings


def new_func():
    """Make it easier to see what is happening in LDA."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)


new_func()

# Reading important files and taking lists (better to work)


def get_df_and_list(file):
    """Take a csv file and return a df and a column as list."""
    df_total = pd.read_csv(file).iloc[:, 1:]
    data_words = df_total.constitution_lemma.to_list()
    return df_total, data_words


df_total, data_words = get_df_and_list('./results/df_total_texts.csv')

# Reading important files and taking lists (better to work)


def get_constitutions_bigrams():
    """Get the constitutions df from csv and take the necessary column."""
    constitutions = pd.read_csv('results/csv/constitutions.csv')

    # Note that the column 'document' has a string that appear as a list.
    # We deal with this small misbehavior by apllying the ast library,
    # ast.literal_eval function.
    data_words_bigrams = constitutions['document'].apply(
        lambda x: ast.literal_eval(x)).to_list()

    return data_words_bigrams


data_words_bigrams = get_constitutions_bigrams()

# Create Dictionary
id2word = corpora.Dictionary(data_words_bigrams)
# id2word.filter_extremes()

# Create Corpus
texts = data_words_bigrams

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Get the number of different words in the corpus
n_words = id2word.num_pos
# Get the most common 50 words in the corpus (possibility removing them as
# stopwords in further runs)
words_list_list = [words.split() for words in data_words]
words_list = [word for words in words_list_list for word in words]
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


# Get the best model or the preferred one
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
    # Get the number of topics in the best model or the one chosen
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

# df_final.columns = ['country', 'year', 'Topic 1', 'Topic 2', 'Topic 3',
#                     'Topic 4', 'Topic 5', 'Topic 6', 'Topic 7', 'Topic 8',
#                     'Topic 9', 'Topic 10', 'Topic 11', 'Topic 12',
#                     'Topic 13', 'Topic 14', 'Topic 15', 'Topic 16',
#                     'Topic 17', 'Topic 18', 'Topic 19', 'Topic 20',
#                     'Topic 21', 'Topic 22', 'Topic 23', 'Topic 24',
#                     'Topic 25', 'Topic 26', 'Topic 27', 'Topic 28',
#                     'Topic 29', 'Topic 30', 'Topic 31', 'Topic 32',
#                     'Topic 33', 'Topic 34', 'Topic 35', 'Topic 36',
#                     'Topic 37', 'Topic 38', 'Topic 39', 'Topic 40',
#                     'Topic 41', 'Topic 42', 'Topic 43', 'Topic 44',
#                     'Topic 45', 'Topic 46', 'Topic 47', 'Topic 48',
#                     'Topic 49', 'Topic 50']

folder = r'results/Topics_models/'
df_final.to_pickle(rf'{folder}/df_topics(50topics_08_05(full_set)).pkl')


vis = gensimvis.prepare(optimal_model, corpus, id2word)
pyLDAvis.enable_notebook()
pyLDAvis.display(vis)
pyLDAvis.save_html(vis, f'{folder}/lda_50topics_all_set(08_05).html')
