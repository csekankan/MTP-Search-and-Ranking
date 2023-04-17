from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator
import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer, util
import torch
from sentence_transformers import CrossEncoder

def train(content_df,model_name):
       if 'curr_model' not in st.session_state:
              st.session_state['curr_model'] = 'sbert'
       st.session_state['curr_model'] = 'sbert'
      
       model = SentenceTransformer(model_name)
       corpus_embeddings = model.encode(content_df.clean.values, convert_to_tensor=True)

       if 'sbert_model' not in st.session_state:
              st.session_state['sbert_model'] = model
       st.session_state['sbert_model'] = model

       if 'sbert_corpus' not in st.session_state:
              st.session_state['sbert_corpus'] = corpus_embeddings
       st.session_state['sbert_corpus'] = corpus_embeddings

       
def cross_score(model_inputs,cross_model):
    scores = cross_model.predict(model_inputs)
    return scores 
def searchwithcrossencoder(query,content_df,itemid):
       if 'curr_model' not in st.session_state:
            return None
       
       if  st.session_state['curr_model'] != 'sbert':
            return None
       
       if 'sbert_corpus' not in st.session_state:
            return None
       
       if 'cross_encoder_model' not in st.session_state:
              cross_model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6', max_length=512)
              st.session_state['cross_encoder_model'] = cross_model
       cross_model=st.session_state['cross_encoder_model']


       model =  st.session_state['sbert_model']
       corpus= st.session_state['sbert_corpus']
       query_embedding = model.encode(query, convert_to_tensor=True)



       alltitles=[]
       correct_hits = util.semantic_search(query_embedding, corpus, top_k=25)[0]
       correct_hits_ids = set([hit['corpus_id'] for hit in correct_hits])
       results =  [content_df.iloc[idx][itemid] for idx in correct_hits_ids]
       newtags=content_df[content_df[itemid].isin(results)]['NewTag']


       model_inputs = [[query,item] for item in newtags]
       scores = cross_score(model_inputs,cross_model)
       #Sort the scores in decreasing order
       ranked_results = [{'Title': str(list(content_df[content_df[itemid]==inp]['title'])[0]), 'Score': score} for inp, score in zip(results, scores)]
       ranked_results = sorted(ranked_results, key=lambda x: x['Score'], reverse=True)
       return ranked_results
      
def search(query,content_df,itemid):
       
       
       if 'curr_model' not in st.session_state:
            return None
       
       if  st.session_state['curr_model'] != 'sbert':
            return None
       
       if 'sbert_corpus' not in st.session_state:
            return None
       
       tokenized_query = query.split(" ")
       model =  st.session_state['sbert_model']
       corpus= st.session_state['sbert_corpus']
      
       query_embedding = model.encode(query, convert_to_tensor=True)
       correct_hits = util.semantic_search(query_embedding, corpus, top_k=10)[0]
       correct_hits_ids = set([hit['corpus_id'] for hit in correct_hits])
       
       results =  [content_df.iloc[idx][itemid[0]] for idx in correct_hits_ids]
 
       # We use cosine-similarity and torch.topk to find the highest 3 scores
       # cos_scores = util.pytorch_cos_sim(query_embedding, corpus)[0]
       # top_results = torch.topk(cos_scores, k=10)
       # recommedations_list=[]
       # print(top_results)
       # for score, idx in zip(top_results[0], top_results[1]):
       #        score = score.cpu().data.numpy() 
       #        idx = idx.cpu().data.numpy()
       #        recommedations_list.append(content_df[[itemid]].iloc[idx][0])
       st.write('Semantic Search Results:')
       return content_df[content_df[itemid[0]].isin(results)][itemid[0]]
