import nltk
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import string
import gensim
from gensim import models
import numpy as np
import wget
import pandas as pd

path = 'C:/Users/linda/OneDrive/Documents/python'
coventry_all_2 = pd.read_csv(path + '/cov_facultyDept.csv')
coventry_all_2['ID'] = [x for x in range(1, len(coventry_all_2.values)+1)]
coventry_all_2.drop(coventry_all_2.columns[0], axis=1, inplace=True)
coventry_all_2.research_field=coventry_all_2.research_field.astype(str) #avoid been seen as a float
#run only first time
#url = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
#wget.download(url)

coventry_processed = coventry_all_2.copy()
coventry_all_2a = coventry_all_2.copy()

def process_string(text):
    text = text.lower() #to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)) #strip punctuation
    return text

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

stop = stopwords.words('english')
lem = WordNetLemmatizer()
def stop_lemmatize(doc):
    tokens = nltk.word_tokenize(doc)
    tmp = ""
    for w in tokens:
        if w not in stop:
            tmp += lem.lemmatize(w, get_wordnet_pos(w)) + " "
    return tmp

def process_string(text):
    text = text.lower() #to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)) #strip punctuation
    text = stop_lemmatize(text)
    return text

def transform_df(df):
    df = df
    df['names'] = df['names'].apply(process_string)
    df['research_field'] = df['research_field'].apply(process_string)
    df['text'] = df['names'] + " " + df['research_field']
    drop_cols = ['names', 'research_field', 'link']
    df = df.drop(drop_cols, axis=1)
    return df

def index_it(single_entry, index):
    words = single_entry.text.split()
    ID = single_entry.ID
    for word in words:
        if word in index.keys():
            index[word].append(ID)
        else:
            index[word] = [ID]
    return index

def index_all(df, index):
    for i in range(len(df)):
        single_entry = df.loc[i,:]
        index = index_it(single_entry = single_entry, index = index)
    return index


def build_index(df, index):
    to_add = transform_df(df)
    index = index_all(df = to_add, index = index)
    return index

idx = build_index(df = coventry_all_2, index = {})


word2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True,  limit=10**5)

def average_vectors(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.vocab]
    if len(doc) == 0:
        return np.zeros(300)
    else:
        return np.mean(word2vec_model[doc], axis=0)

def prepare_ranking(df):
    corpus = df[['ID', 'text']].copy()
    doc_vecs = {}
    for i in range(len(corpus)):
        row = corpus.loc[i,:]
        text = row.text.split()
        doc_vecs[row.ID]=average_vectors(word2vec, text)
    doc_vecs = pd.DataFrame.from_dict(data=doc_vecs, orient="index")
    doc_vecs['ID'] = doc_vecs.index
    return doc_vecs

doc_vecs = prepare_ranking(df=coventry_all_2)

def process_query(query):
    norm = process_string(query)
    return norm.split()

def lists_intersection(lists):
    intersect = list(set.intersection(*map(set, lists)))
    intersect.sort()
    return intersect

def search_googleish(query, index=idx):
    query_split = process_query(query)
    retrieved = []
    for word in query_split:
        if word in index.keys():
            retrieved.append(index[word])
    if len(retrieved)>0:
        result = lists_intersection(retrieved)
    else:
        result = ['No Information Found']
    return result

meta = coventry_all_2a.copy()

def connect_id_df(retrieved_id, df):
    return df[df.ID.isin(retrieved_id)].reset_index(drop=True)


def cos_similarity(a, b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    return(cos)


def rank_results(query, results):
    query_norm = process_query(query)
    query_vec = average_vectors(word2vec, query_norm)
    result_vecs = connect_id_df(results.ID, doc_vecs)
    cos_sim = []
    for i in range(len(result_vecs)):
        doc_vec = result_vecs.loc[i,:].drop(['ID'])
        cos_sim.append(cos_similarity(doc_vec, query_vec))
    results['rank'] = cos_sim
    results = results.sort_values('rank', axis=0)
    return results


def print_results(result_df):
    for i in range(len(result_df)):
        res = result_df.loc[i, :]
        print( res.names)
        print("Research Field: ", res.research_field)
        print("Research_Interest: ",res.research_interest )
        print("Subfaculty: ", res.subfaculty)
        print("Faculty: ", res.subfaculty)
        print("CU link: ", res.link_CU)
        if i == len(result_df):
            print("Scholar link:", res.link)
        else:
            print("{}\n" .format(res.link))
            print("------------------------------------")


def search(query, dat=None):
    result = search_googleish(query)
    result = connect_id_df(result, meta)
    result = rank_results(query, result)
    print_results(result)

'''
query = input("Search for:")
print('*******************')
search(query)
'''