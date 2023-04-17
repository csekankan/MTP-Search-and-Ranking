
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

def file_up(separator):
      df=pd.DataFrame()
      w = st.file_uploader("Upload a data")
      
      if w is not None:
              
               if(separator=='\\t'):
                      df = pd.read_csv(w,sep='\t')
               else:
                    df = pd.read_csv(w,sep=separator)
                 
     
     
      return df
def run():
      
      text_cols = st.text_input("Enter columns contain texts, separated by comma(,)", value="")        
      search_result_col = st.text_input("Enter column(single) you want to show in search results", value="")
      fraction = st.text_input("fraction of data you want to process", value="1")
      separator = st.selectbox(
                 f"Data separator",
                  [',','|' , '\\t']
                  )  
      df= pd.DataFrame()
      text_cols=text_cols.strip().split(",")
      search_result_col=search_result_col.strip().split(",")
      fraction=float(fraction)
      search_result_col=[x.strip() for x in search_result_col]
      text_cols=[x.strip() for x in text_cols]
    
      
      df=file_up(separator)
      cols=[]
     
    
      for i in  text_cols:
            if(len(i)>0):
             cols.append(i)
      for i in  search_result_col:
            if(len(i)>0):
             cols.append(i)
      cols=list(set(cols))      

      if df is not None and df.shape[1]>0  and cols is not None and len(cols)>0:
        df=df[cols]
   
   
     
      
      
      return df,text_cols,search_result_col
         

    	
   
    	



