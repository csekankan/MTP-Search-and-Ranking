import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


import page
from page import dataLoader,preprocess

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import os
import pathlib
from os import listdir
from os.path import isfile, join



"""
# Search and Ranking
"""


     
if __name__ == "__main__":
    
    df,text_cols,search_result_col=dataLoader.run()
   
    preprocess.construct_sidebar(df,text_cols,search_result_col)
