
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder ,MinMaxScaler
from page import tfidf,bm25,sbert
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

from nltk.stem import WordNetLemmatizer
stopwords = nltk.corpus.stopwords.words('english')
from nltk.tokenize import word_tokenize
import re

def clean_text(text):
    
    text = text.lower()  # lowercase text
    # replace the matched string with ' '
    text = re.sub( re.compile("\'s"), ' ', text)
    text = re.sub(re.compile("\\r\\n"), ' ', text)
    text = re.sub(re.compile(r"[^\w\s]"), ' ', text)
    return text


def stopwprds_removal_gensim_custom(str):
  
    text_tokens = word_tokenize(str)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords]
    str_t = " ".join(tokens_without_sw)

    return str_t
def data_cleanup(content_df,features):

  
     content_df['NewTag']=""

     for i in features:
       content_df[i] = content_df[i].fillna(' ')
     for i in features:
       content_df['NewTag']+=(' '+content_df[i])
     content_df['NewTag']=content_df['NewTag'].astype(str)
     content_df['clean'] = content_df['NewTag'].apply(clean_text)
     content_df['clean'] = content_df['clean'].apply(stopwprds_removal_gensim_custom) 

   
    
     return content_df



def construct_sidebar(df,text_cols,search_results_col):
        if(df is not None and df.shape[0]>0):
               df=data_cleanup(df,text_cols)
      
        st.sidebar.markdown(
            '<p class="header-style"><h2>Model Training</h2></p>',
            unsafe_allow_html=True
        )
        if df is None or df.shape[0]==0:
             return 

        df = df.fillna(' ')
        
              
        model_selected = st.sidebar.selectbox(
            f"Search Type",
           ['Keyword based Search','Symmetric Semantic search','Asymmetric Semantic search']
        )

        algo_selected=None
        if(model_selected in ['Keyword based Search']):
           algo_selected = st.sidebar.selectbox(
                       f"Select Model",
                      ['TF-IDF','BM-25']
                      )

        if(model_selected in ['Symmetric Semantic search']):
           algo_selected = st.sidebar.selectbox(
                       f"Select Pretrained Models",
                      ['multi-qa-MiniLM-L6-cos-v1','paraphrase-multilingual-mpnet-base-v2','paraphrase-albert-small-v2'
                       ,'paraphrase-multilingual-MiniLM-L12-v2','paraphrase-MiniLM-L3-v2','distiluse-base-multilingual-cased-v2'
                       ,'distiluse-base-multilingual-cased-v1']
                      )
           

        if(model_selected in ['Asymmetric Semantic search']):
           algo_selected = st.sidebar.selectbox(
                       f"Select Pretrained Models",
                      ['msmarco-MiniLM-L-6-v3','msmarco-MiniLM-L-12-v3','msmarco-distilbert-base-v3'
                       ,'msmarco-distilbert-base-v4','msmarco-roberta-base-v3','msmarco-distilbert-base-dot-prod-v3'
                       ,'msmarco-roberta-base-ance-firstp','msmarco-distilbert-base-tas-b'
                       ]
                      )
          #  otherlanguage = st.sidebar.checkbox('Multilingual/ non-english language')

          #  if otherlanguage:
          #   st.write('Great!')
        
        if st.sidebar.button('Train')  :
    
            if(algo_selected=='TF-IDF'):
                  tfidf.train(df)
            if(algo_selected=='BM-25'):
                  bm25.train(df)
            if(algo_selected in   ['msmarco-MiniLM-L-6-v3','msmarco-MiniLM-L-12-v3','msmarco-distilbert-base-v3'
                       ,'msmarco-distilbert-base-v4','msmarco-roberta-base-v3','msmarco-distilbert-base-dot-prod-v3'
                       ,'msmarco-roberta-base-ance-firstp','msmarco-distilbert-base-tas-b'
                       ]):
                sbert.train(df,algo_selected)
            if(algo_selected in   ['multi-qa-MiniLM-L6-cos-v1','paraphrase-multilingual-mpnet-base-v2','paraphrase-albert-small-v2'
                       ,'paraphrase-multilingual-MiniLM-L12-v2','paraphrase-MiniLM-L3-v2','distiluse-base-multilingual-cased-v2'
                       ,'distiluse-base-multilingual-cased-v1']):
                sbert.train(df,algo_selected)
                


        txt = st.sidebar.text_area('Search Text','')
        if st.sidebar.button('Search'):
          if(len(txt.strip())>0):
               
              if 'curr_model'  in st.session_state:
                  
                  if st.session_state['curr_model'] == 'tf-idf':
                      st.write(tfidf.search(txt,df,search_results_col))
                  if st.session_state['curr_model'] == 'bm25':  
                      st.write(bm25.search(txt,df,search_results_col))
                  if st.session_state['curr_model'] == 'sbert':  
                      st.write(sbert.search(txt,df,search_results_col))

        return df
        
        