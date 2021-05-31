import numpy as np
import pandas as pd
import csv
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import requests
import codecs
from bs4 import BeautifulSoup
from collections import namedtuple
import pickle
import hyperopt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, CatBoostClassifier
import catboost
import hyperopt

WebDocInfo = namedtuple("WebDocInfo", ["doc_id", "is_redirect", "is_accesible"])

def get_url(doc_id):
    path='content/'
    filename = "{}.dat".format(doc_id)
    with codecs.open(path + filename, 'r', 'utf-8') as f:
        url = f.readline().strip()
    return url

def get_web_doc_info(doc_id):
    is_redirect = False
    is_accessible = True
    scheme = "http"
    url = '{0}://{1}/'.format(scheme, get_url(doc_id))
    try:
        response = requests.head(url, timeout=1)
        is_redirect = response.is_redirect
    except Exception as e:
        is_accessible = False
   
    return WebDocInfo(doc_id, is_redirect, is_accessible)

def web_doc_info_process(doc_id, with_lock=True):
    web_doc_info = get_web_doc_info(doc_id)
    if with_lock:
        with lock:
            dict_web_doc_info[doc_id] = web_doc_info
            pbar.update(1)
    else:
        dict_web_doc_info[doc_id] = web_doc_info

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_body(doc_id):
    with open("content/{}.dat".format(doc_id), encoding='UTF-8') as f:
        url = f.readline().strip()
        soup = BeautifulSoup(f, 'lxml')
    if soup.body is None:
        return ''
    else:
        return soup.body.get_text(" ").lower()

def get_preprocessed_text(doc_id):
    title = doc_titles[doc_id].lower()
    body = get_body(doc_id)
    text = title + " " + body
    stems = [stemmer_rus.stem(word) for word in re.sub('[^a-zа-я0-9]', ' ', text).split()]
    #if not word in stop_words
    preprocessed_text = ' '
    preprocessed_text = preprocessed_text.join(stems)
    return preprocessed_text

def get_preprocessed_text_wo_stop_words(doc_id):
    text = dict_doc_text[doc_id]
    stems = [stem for stem in text.split() if not stem in stop_words]
    preprocessed_text = ' '
    preprocessed_text = preprocessed_text.join(stems)
    return preprocessed_text

def doc_text_process(doc_id, with_lock=True):
    doc_text = get_preprocessed_text(doc_id)
    if with_lock:
        with lock:
            if doc_id not in dict_doc_text:
                dict_doc_text[doc_id] = doc_text
            pbar.update(1)
    else:
        dict_doc_text[doc_id] = doc_text

def doc_text_process_wo_stop_words(doc_id, with_lock=True):
    doc_text = get_preprocessed_text_wo_stop_words(doc_id)
    if with_lock:
        with lock:
            if doc_id not in dict_doc_text_wo_stop_words:
                dict_doc_text_wo_stop_words[doc_id] = doc_text
            pbar.update(1)
    else:
        dict_doc_text_wo_stop_words[doc_id] = doc_text

def doc_text_process_safe(doc_id, with_lock=True):
    try:
        doc_text_process(doc_id, with_lock)
    except:
        print(doc_id)

def doc_text_process_wo_stop_words_safe(doc_id, with_lock=True):
    try:
        doc_text_process_wo_stop_words(doc_id, with_lock)
    except:
        print(doc_id)

def get_preprocessed_title(doc_id):
    text = doc_titles[doc_id].lower()
    stems = [stemmer_rus.stem(word) for word in re.sub('[^a-zа-я0-9]', ' ', text).split()
             if not word in stop_words]
    preprocessed_text = ' '
    preprocessed_text = preprocessed_text.join(stems)
    return preprocessed_text

def doc_title_process(doc_id, with_lock=True):
    doc_title = get_preprocessed_title(doc_id)
    if with_lock:
        with lock:
            if doc_id not in doc_titles_processed:
                doc_titles_processed[doc_id] = doc_title
            pbar.update(1)
    else:
        doc_titles_processed[doc_id] = doc_title

def get_content_top_words(doc_id, dict_doc_text_wo_stop_words, n_top=10):
    doc_text = dict_doc_text_wo_stop_words[doc_id]
    #doc_text = doc_text + " " + doc_titles_processed[doc_id]
    if doc_text != '':
        vec = CountVectorizer().fit([doc_text])
        doc_csr = vec.transform([doc_text])
        doc_arr = doc_csr.toarray().ravel()
        words_freq = [(word, doc_arr[idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:n_top]
    else:
        return []

def get_corpus(doc_ids, dict_doc_text_wo_stop_words):
    corpus = {}
    for doc_id in tqdm(doc_ids):
        try:
            content = ' '
            title = doc_titles_processed[doc_id]
            content = content.join([w for w, _ in get_content_top_words(doc_id, dict_doc_text_wo_stop_words)])
            corpus[doc_id] = content + " " + title
        except:
            print(doc_id)
            break;
    return corpus

def add_to_df_web_info(df, dict_web_doc_info):
    is_redirect_col = []
    is_accessible_col = []
    for doc_id in df.doc_id:
        is_redirect_col.append(str(int(dict_web_doc_info[doc_id].is_redirect)))
        is_accessible_col.append(str(int(dict_web_doc_info[doc_id].is_accesible)))
    df['is_redirect'] = is_redirect_col
    df['is_accessible'] = is_accessible_col

def fill_grp_20(grp_indices, X_vec):
    n_features = 20
    X_train_grp = np.empty(shape=(grp_indices.size, n_features), dtype=np.float)
    for i, all_dist in enumerate(pairwise_distances(X_vec[grp_indices], metric='cosine')):
        X_train_grp[i, :n_features] = sorted(all_dist)[1:n_features + 1]
    return X_train_grp

def fill_grp_80(grp_indices, X_vec):
    n_features = 20
    X_train_grp = np.empty(shape=(grp_indices.size, n_features * 4), dtype=np.float)
    for i, all_dist in enumerate(pairwise_distances(X_vec[grp_indices], metric='cosine')):
        X_train_grp[i, :n_features] = sorted(all_dist)[1:n_features + 1]
    X_train_grp[:, n_features:2*n_features] = np.mean(X_train_grp[:, :n_features], axis=0)
    X_train_grp[:, 2*n_features:3*n_features] = np.std(X_train_grp[:, :n_features], axis=0)
    X_train_grp[:, 3*n_features:] = np.median(X_train_grp[:, :n_features], axis=0)
    return X_train_grp

def get_X_20(df, corpus):
    doc_ids  = df.doc_id.drop_duplicates()
    vect_tfidf = TfidfVectorizer()          
    corpus_tfidf_csr = vect_tfidf.fit_transform(corpus.values())
    corpus_tfidf = corpus_tfidf_csr.toarray()

    corpus_vec = {}
    for idx, t in enumerate(corpus.items()):
        corpus_vec[t[0]] = corpus_tfidf[idx]
    
    X_vec = []
    for idx in df.index:
        doc_id = df.iloc[idx].doc_id
        X_vec.append(list(corpus_vec[doc_id]))
    X_vec = np.asarray(X_vec)
    
    n_features = 20
    X_train = np.empty(shape=(df.shape[0], n_features), dtype=np.float)
    df_grouped = df.groupby('group_id')
    i = 0
    for grp_id, grp_indices in df_grouped.groups.items():
        j = i + grp_indices.size
        X_train[i:j] = fill_grp_20(grp_indices, X_vec)
        i = j
        
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
        
    X_train_web = []
    for idx in df.index:
        doc_id = df.iloc[idx].doc_id
        doc_features = list(X_train_s[idx])
        doc_features.append(df.iloc[idx].is_redirect)
        doc_features.append(df.iloc[idx].is_accessible)
        X_train_web.append(doc_features)
    return (X_train_web, X_train_s, X_train)

def get_X_80(df, corpus):
    doc_ids  = df.doc_id.drop_duplicates()
    vect_tfidf = TfidfVectorizer()          
    corpus_tfidf_csr = vect_tfidf.fit_transform(corpus.values())
    corpus_tfidf = corpus_tfidf_csr.toarray()

    corpus_vec = {}
    for idx, t in enumerate(corpus.items()):
        corpus_vec[t[0]] = corpus_tfidf[idx]
    
    X_vec = []
    for idx in df.index:
        doc_id = df.iloc[idx].doc_id
        X_vec.append(list(corpus_vec[doc_id]))
    X_vec = np.asarray(X_vec)
    
    n_features = 20
    X_train = np.empty(shape=(df.shape[0], n_features * 4), dtype=np.float)
    df_grouped = df.groupby('group_id')
    i = 0
    for grp_id, grp_indices in df_grouped.groups.items():
        j = i + grp_indices.size
        X_train[i:j] = fill_grp_80(grp_indices, X_vec)
        i = j
        
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
        
    X_train_web = []
    for idx in df.index:
        doc_id = df.iloc[idx].doc_id
        doc_features = list(X_train_s[idx])
        doc_features.append(df.iloc[idx].is_redirect)
        doc_features.append(df.iloc[idx].is_accessible)
        X_train_web.append(doc_features)
    return (X_train_web, X_train_s, X_train)

def fill_grp_80_simple(grp_indices, df, corpus):
    sub_corpus = {}
    doc_ids = df.iloc[grp_indices].doc_id
    for doc_id in doc_ids:
        sub_corpus[doc_id] = corpus[doc_id]
    
    vect = TfidfVectorizer()          
    sub_corpus_csr = vect.fit_transform(sub_corpus.values())
    sub_corpus_arr = sub_corpus_csr.toarray()
    
    sub_corpus_vec = {}
    for idx, t in enumerate(sub_corpus.items()):
        sub_corpus_vec[t[0]] = sub_corpus_arr[idx]
    
    X_vec = []
    for idx in grp_indices:
        doc_id = df.iloc[idx].doc_id
        X_vec.append(list(sub_corpus_vec[doc_id]))
    X_vec = np.asarray(X_vec)
    
    n_features = 20
    X_train_grp = np.empty(shape=(grp_indices.size, n_features * 4), dtype=np.float)
    for i, all_dist in enumerate(pairwise_distances(X_vec, metric='cosine')):
        X_train_grp[i, :n_features] = sorted(all_dist)[1:n_features + 1]
    X_train_grp[:, n_features:2*n_features] = np.mean(X_train_grp[:, :n_features], axis=0)
    X_train_grp[:, 2*n_features:3*n_features] = np.std(X_train_grp[:, :n_features], axis=0)
    X_train_grp[:, 3*n_features:] = np.median(X_train_grp[:, :n_features], axis=0)
    return X_train_grp

def get_X_80_simple(df, doc_ids, dict_doc_text_wo_stop_words, corpus):
    n_features = 20
    X_train = np.empty(shape=(df.shape[0], n_features * 4), dtype=np.float)
    df_grouped = df.groupby('group_id')
    i = 0
    for grp_id, grp_indices in df_grouped.groups.items():
        j = i + grp_indices.size
        X_train[i:j] = fill_grp_80_simple(grp_indices, df, corpus)
        i = j
        
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
        
    X_train_web = []
    for idx in df.index:
        doc_id = df.iloc[idx].doc_id
        doc_features = list(X_train_s[idx])
        doc_features.append(df.iloc[idx].is_redirect)
        doc_features.append(df.iloc[idx].is_accessible)
        X_train_web.append(doc_features)
    return (X_train_web, X_train_s, X_train)

def get_X_web(X_train, df, indices):
    X_train_web = []
    for idx in indices:
        doc_id = df.iloc[idx].doc_id
        doc_features = list(X_train[idx])
        doc_features.append(df.iloc[idx].is_redirect)
        doc_features.append(df.iloc[idx].is_accessible)
        X_train_web.append(doc_features)
    return X_train_web

def get_t_val_indices(df, grp_val_indices):
    grp_t_indices = []
    for group_id in range(1, 130):
        if group_id not in set(grp_val_indices):
            grp_t_indices.append(group_id)
        
    df_grouped = df.groupby("group_id")

    val_indices = []
    t_indices = []
    for grp_id, grp_indices in df_grouped.groups.items():
        if grp_id in set(grp_val_indices):
            val_indices.extend(grp_indices)
        if grp_id in set(grp_t_indices):
            t_indices.extend(grp_indices)
    return t_indices, val_indices


