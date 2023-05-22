import pandas as pd #Data maniuplation
import numpy as np #Data manipulation

from sklearn.model_selection import train_test_split #Splitting data into train and test samples
from sklearn.metrics import classification_report #Model evaluation metrics
from sklearn import tree #Decision metrics

import plotly.express as px #Data visualization
import plotly.graph_objects as go #Data visualization
import graphviz #Plotting decision tree graphs


import os
import sys

class CART:
    def __init__(self, csvFilePath : str) -> None:
        pd.options.display.max_columns = 50

        self.file = pd.read_csv(csvFilePath, encoding= 'utf-8')

        print(self.file)

    def execution(self) -> None:
        input = self.file[['qPa', 'Pulso', 'Resp']]
        output = self.file['Gravidade'].values



    # def fitting()