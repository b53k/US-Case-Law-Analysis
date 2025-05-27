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

class TextSearch:
    def __init__(self, df_file_path, embedding_type):
        print('Initializing searcher.....')
        self.embedding_type = embedding_type
        self.df = pd.read_parquet(df_file_path, engine='pyarrow')
        self.search_index = faiss.read_index(f'{embedding_type}_search_index.bin')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if embedding_type=='bge':
            self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
            self.model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5").to(self.device)


        elif embedding_type=='legalbert':
            self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
            self.model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased").to(self.device)


        elif embedding_type=='doc2vec':
            self.model = Doc2Vec.load("doc2vec_model.model")


        elif embedding_type=='BOW':
            with open("BOW_vectorizer.pkl", "rb") as f:
                self.model = pickle.load(f)

            with open("BOW_svd.pkl", "rb") as f:
                self.svd = pickle.load(f)


        elif embedding_type=='TFIDF':
            with open("TFIDF_vectorizer.pkl", "rb") as f:
                self.model = pickle.load(f)
                
            with open("TFIDF_svd.pkl", "rb") as f:
                self.svd = pickle.load(f)


        print('Searcher is ready.')

    def get_query_embedding(self, query):
        if self.embedding_type=='bge' or self.embedding_type=='legalbert':
            # Tokenize input
            inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True).to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use the [CLS] token embedding (pooled output for classification)
            query_embedding = outputs.last_hidden_state[:, 0, :]  # shape: [batch_size, hidden_size]
            query_embedding = query_embedding.cpu().numpy()


        elif self.embedding_type=='BOW' or self.embedding_type=='TFIDF':
            vec = self.model.transform([query])
            query_embedding = self.svd.transform(vec).astype('float32')
        
        elif self.embedding_type=='doc2vec':
            query_embedding = self.model.infer_vector(query.split())
            query_embedding = query_embedding[np.newaxis, :]

        else:
            raise TypeError(f'Wrong embedding type \"{self.embedding_type}\"')

        return query_embedding



    def search(self, query, k):
        print('searching...')
        query_embedding = self.get_query_embedding(query)
        distances, indices = self.search_index.search(query_embedding, k)

        return self.df.iloc[indices[0]]
