import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm  # For progress tracking
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import faiss

import pickle

import argparse



def generate_embeddings_batch(texts, tokenizer, model, device):
    # Tokenize batch and truncate to 512 tokens
    inputs = tokenizer(texts, truncation=True, max_length=512, return_tensors="pt", padding=True)
    
    # Move inputs to the correct device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Compute embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token embedding
    
    return embeddings




###################  for bge embedding  #######################
def bge_embedding(df, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5").to(device)
    
    texts = df['text'].tolist()
    embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = generate_embeddings_batch(batch_texts, tokenizer, model, device)
        embeddings.extend(batch_embeddings)

    
    embeddings_array = np.stack(embeddings, axis=0)
    np.save('bge_embedding.npy', embeddings)
    
    return df

###################  for legalbert of words embedding  #######################
def legalbert_embedding(df, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased").to(device)
    
    texts = df['text'].tolist()
    embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = generate_embeddings_batch(batch_texts, tokenizer, model, device)
        embeddings.extend(batch_embeddings)

    
    embeddings_array = np.stack(embeddings, axis=0)
    np.save('legalbert_embedding.npy', embeddings)
    
    return df




###################  for bag of words embedding  #######################
def BOW_embedding(df):
    L = list(df['text'])
    print('processing BOW embedding ...')
    vectorizer = CountVectorizer(max_features=10000)
    X_sparse = vectorizer.fit_transform(L)
    
    svd = TruncatedSVD(n_components=800, random_state=42)  
    X_dense = svd.fit_transform(X_sparse)
    del X_sparse
    np.save("BOW_embedding.npy", X_dense)
    with open("BOW_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("BOW_svd.pkl", "wb") as f:
        pickle.dump(svd, f)

###################  for TFIDF embedding  #######################

def TFIDF_embedding(df):
    L = list(df['text'])
    print('processing TFIDF embedding ...')
    vectorizer = TfidfVectorizer(max_features=10000)
    X_tfidf = vectorizer.fit_transform(L)

    svd = TruncatedSVD(n_components=800, random_state=42)
    X_dense = svd.fit_transform(X_tfidf) 
    del X_tfidf
    np.save("TFIDF_embedding.npy", X_dense)
    with open("TFIDF_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("TFIDF_svd.pkl", "wb") as f:
        pickle.dump(svd, f)


###################  for doc2vec embedding  #######################

def doc2vec_embedding(df):
    corpus = list(df['text'])

    # Tag each document with a unique ID
    tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(corpus)]


    model = Doc2Vec(vector_size=800, window=5, min_count=2, workers=4, epochs=40)

    print('Training the doc2vec model ...')

    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    model.save("doc2vec_model.model")




    
    

def create_search_index(embedding_type):
    if embedding_type=='doc2vec':
        X = np.load('doc2vec_model.model.dv.vectors.npy')
        
    else:
        X = np.load(f'{embedding_type}_embedding.npy') 
    index = faiss.IndexFlatL2(X.shape[1])  # L2 distance
    index.add(X)
    file_name = f'{embedding_type}_search_index.bin' 
    faiss.write_index(index, file_name)
    print(f'search index saved as {file_name}')



def main(df_file_path, embedding_type):
    

    if embedding_type=='bge':
        df = pd.read_parquet(df_file_path, engine='pyarrow')
        bge_embedding(df)
    elif embedding_type=='legalbert':
        df = pd.read_parquet(df_file_path, engine='pyarrow')
        legalbert_embedding(df)
    elif embedding_type=='BOW':
        df = pd.read_parquet(df_file_path, engine='pyarrow')
        BOW_embedding(df)
    elif embedding_type=='TFIDF':
        df = pd.read_parquet(df_file_path, engine='pyarrow')
        TFIDF_embedding(df)
    elif embedding_type=='doc2vec':
        df = pd.read_parquet(df_file_path, engine='pyarrow')
        doc2vec_embedding(df)
    else:
        raise KeyError(f'Invalid Embedding Type \"{embedding_type}\"')
    
    create_search_index(embedding_type)








if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Pass arguments to main function")

    parser.add_argument("df_file_path", type=str, help="File path of the parquet file")
    parser.add_argument("embedding_type", type=str, help="method used for the embedding")

    args = parser.parse_args()


    main(df_file_path=args.df_file_path, embedding_type=args.embedding_type)
