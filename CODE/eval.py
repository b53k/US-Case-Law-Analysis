import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import faiss


import pickle

import argparse
from bert_score import score

from search import TextSearch



if __name__=='__main__':
    searcher=None

    emb_type_list = ['bge', 'legalbert', 'BOW', 'TFIDF', 'doc2vec']
    df_file_path = 'ga_final.parquet'

    query_df = pd.read_parquet('ga_final.parquet', engine='pyarrow')
    query_list = list(query_df['text'])[:50]
    

    k=5
    result_dict={}


    for embedding_type in emb_type_list:
        del searcher
        searcher=TextSearch(df_file_path, embedding_type)
        search_result_list=[]
        query_list_for_score=[]
        for query in tqdm(query_list):
            search_result = searcher.search(query, k)
            search_result_list+=list(search_result['text'])
            query_list_for_score+=[query]*k


        P, R, F1 = score(query_list_for_score, search_result_list, lang="en", verbose=True)
        F1=F1.view(-1, k)
        
        result_dict[embedding_type] = torch.mean(F1, dim=0)
    print(result_dict)
