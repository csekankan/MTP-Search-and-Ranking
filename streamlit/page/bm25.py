from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator
import streamlit as st
import pickle
from rank_bm25 import *


def train(content_df):
       if 'curr_model' not in st.session_state:
              st.session_state['curr_model'] = 'bm25'
       st.session_state['curr_model'] = 'bm25'
      
       tokenized_corpus = [doc.split(" ") for doc in content_df['clean'].values]       # dictionary = corpora.Dictionary(texts)
       # corpus = [dictionary.doc2bow(text) for text in texts]
       bm25_obj  = BM25Okapi(tokenized_corpus)
       if 'bm25_model' not in st.session_state:
              st.session_state['bm25_model'] = bm25_obj
       st.session_state['bm25_model'] = bm25_obj

       if 'bm25_corpus' not in st.session_state:
              st.session_state['bm25_corpus'] = tokenized_corpus
       st.session_state['bm25_corpus'] = tokenized_corpus

       
 

def search(query,content_df,itemid):
       
       
       if 'curr_model' not in st.session_state:
            return None
       
       if  st.session_state['curr_model'] != 'bm25':
            return None
       
       if 'bm25_corpus' not in st.session_state:
            return None
       
       tokenized_query = query.split(" ")
       bm25 =  st.session_state['bm25_model']
       text_token= st.session_state['bm25_corpus']
      
       doc_scores = bm25.get_scores(tokenized_query)
       docs = bm25.get_top_n(query=tokenized_query, documents=content_df['clean'].values, n=10)
       df_search = content_df[content_df["clean"].isin(docs)][itemid[0]]
      
       st.write('BM25 Search Results:')
       return df_search
