from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator
import streamlit as st
import pickle
import gensim


def train(content_df):
       if 'curr_model' not in st.session_state:
              st.session_state['curr_model'] = 'tf-idf'
       st.session_state['curr_model'] = 'tf-idf'
       # Intializating the tfIdf model
       tfidf = TfidfVectorizer(stop_words='english')
       # Fit the TfIdf model
       tfidf_tran=tfidf.fit_transform(content_df.clean)

       if 'tfidf_model' not in st.session_state:
              st.session_state['tfidf_model'] = tfidf
       st.session_state['tfidf_model'] = tfidf


       if 'tfidf_token' not in st.session_state:
              st.session_state['tfidf_token'] = tfidf_tran
       st.session_state['tfidf_token'] = tfidf_tran
       # with open('tf-idf.pkl','wb') as handle:
       #        pickle.dump(tfidf_tran, handle)
       # with open('tf-idf_model.pkl','wb') as handle:
       #        pickle.dump(tfidf, handle)
 
 

def search(query,content_df,itemid):

       print('aaaaaaaaaaaaaaaaaaaaa'+str(st.session_state['curr_model'] ))
       # print( st.session_state['curr_model'])
       tfidf = TfidfVectorizer(stop_words='english')
       if 'tfidf_model' not in st.session_state:
             return None
       if 'tfidf_token' not in st.session_state:
             return None
       
       tfidf = st.session_state['tfidf_model']
       #pickle.load(open('tf-idf_model.pkl','rb')) 

       if 'curr_model' not in st.session_state:
            return None
       
       if  st.session_state['curr_model'] != 'tf-idf':
            return None

       tfidf_tran = st.session_state['tfidf_token']
       #pickle.load(open('tf-idf.pkl','rb'))
       query_vec = tfidf.transform([query]) # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
       results = cosine_similarity(tfidf_tran,query_vec).reshape((-1,)) # Op -- (n_docs,1) -- Cosine Sim with each doc
       # Print Top 10 results
       ids=[]
       for i in results.argsort()[-10:][::-1]:
      
         ids.append( content_df.iloc[i][itemid][0])
       st.write('tf-idf Search Results:')
       return content_df[content_df[itemid[0]].isin(ids)][itemid]
