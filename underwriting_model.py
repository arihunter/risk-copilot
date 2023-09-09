# do in 1 hr
import pandas as pd
import numpy as np
from sklearn import ensemble
from typing import List
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text


class UnderWriter():

  def __init__(self,threshold:int):
    #self.df = df
    self.threshold = threshold

  def extract_important_features(self) -> np.ndarray:

    """This function returns a list of scores which depict the importance of that variable in predicting the target variable."""

    #create test and train dataset
    #df = pd.read_csv(path)
    df_curr = self.df
    df_curr.drop(["address","slno"],inplace=True,axis = 1)

    df_train = df_curr[df_curr["split_set"]=="train"]
    df_test = df_curr[df_curr["split_set"]=="holdout"]

    df_train.drop(["split_set"],axis=1,inplace=True)
    df_test.drop(["split_set"],axis=1,inplace=True)
    
    #using train part only from now onwards
    #dropping all the Nan
    df_train.dropna(inplace=True)
    labels = df_train["dep_var"]
    df_train.drop(["dep_var"],axis=1,inplace=True)
    
    #creating data for decision tree
    features = df_train
    gbm = ensemble.GradientBoostingRegressor()
    gbm.fit(features,labels)

    feature_importance = gbm.feature_importances_
    return feature_importance,df_train,labels


  def create_dataset_for_underwriter_model(self,feature_importance:np.ndarray,df_train:pd.DataFrame) -> pd.DataFrame:
    
    """This function creates dataset for the underwriting model."""

    score_with_idx = []
    for idx,score in enumerate(feature_importance):
      score_with_idx.append((score,idx))
    sortedFeatures = sorted(score_with_idx)
    sortedFeatures.reverse()

    currVal = 0
    i = 0
    while(currVal<self.threshold):
      currVal = currVal+sortedFeatures[i][0]
      i = i+1 
    sortedFeatures = sortedFeatures[:i]
    sortedIdx = [x[1] for x in sortedFeatures]
    df_train_dt = df_train.iloc[:,sortedIdx]
    return df_train_dt



  # def create_underwriting_model(self) -> str:
  #   """This function returns the underwriting model for a given dataframe"""

  #   feature_importance,df_train,labels = self.extract_important_features()
  #   underwriting_model_dataset = self.create_dataset_for_underwriter_model(feature_importance,df_train)

    
  #   dtree = DecisionTreeClassifier()
  #   dtree.fit(underwriting_model_dataset, labels)
  #   feature_names = list(underwriting_model_dataset.columns)
    
    
  #   underwriting_model = export_text(dtree, feature_names=feature_names)
  #   return underwriting_model

    def create_underwriting_model(self,path:str) -> str:
      
      """Creates underwriting model for the given data. This underwriting model will further be used to asess risk of the lending portfolio". The output of the function is a string."""
      df = pd.read_csv(path)
      df.drop(["address","slno"],inplace=True,axis = 1)

      df_train = df[df["split_set"]=="train"]
      df_test = df[df["split_set"]=="holdout"]

      df_train.drop(["split_set"],axis=1,inplace=True)
      df_test.drop(["split_set"],axis=1,inplace=True)
      
      
      df_train.dropna(inplace=True)
      labels = df_train["dep_var"]
      df_train.drop(["dep_var"],axis=1,inplace=True)
      
      #creating data for decision tree
      features = df_train
      gbm = ensemble.GradientBoostingRegressor()
      gbm.fit(features,labels)

      feature_importance = gbm.feature_importances_

      score_with_idx = []
      for idx,score in enumerate(feature_importance):
        score_with_idx.append((score,idx))
      sortedFeatures = sorted(score_with_idx)
      sortedFeatures.reverse()
      
      currVal = 0
      i = 0
      while(currVal<self.threshold):
        currVal = currVal+sortedFeatures[i][0]
        i = i+1 
      sortedFeatures = sortedFeatures[:i]
      sortedIdx = [x[1] for x in sortedFeatures]
      df_train_dt = df_train.iloc[:,sortedIdx]


      #create and fit decision tree
      dtree = DecisionTreeClassifier()
      dtree.fit(df_train_dt, labels)
      feature_names = list(df_train_dt.columns)
      

      #create underwriting model
      underwriting_model = export_text(dtree, feature_names=feature_names)
      print(underwriting_model)
      return underwriting_model

