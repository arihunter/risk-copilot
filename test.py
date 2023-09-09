import pandas as pd 
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier


def risk_profiling(start_dt:str,end_dt:str,query:str,col_name:str="bureau_score") -> str:
  """
  This function is used to do risk profiling for borrowers using a indicator, for a given time period. Here the indicator could be the following:

  bureau score: This score is an indicator of the credit-worthiness of a borrower. People who do not have a bureau score or have a blank bureau score are called New-To-Credit or NTC. NTC are customers who have not taken a loan before and do not have any record/history in credit bureau. Similarly, people who have a bureau score or have taken a loan before are referred to as non-NTC. Using the bureau score the function can calculate the following:
      ntc_fraction which is the fraction of NTC in the portfolio
      non_ntc_fraction which is the fraction of non-ntc in the portfolio
      ntc_npa which is the bad-rate or npa of NTC in the portfolio
      non_ntc_npa which is the bad-rate or npa of non-NTC in the portfolio
  """
  # start_dt = st.text_input("Start Date")
  # end_dt = st.text_input("End Date")
  dataset_idx=[0,1]
  for idx in dataset_idx:
    if st.session_state[dataset_keys[idx]] == False:
      return "Sufficient data not available !, please provide all the required data."

  #sanity check
  # dateCheckPrompt = "I will provide you with a user query , you have to analyse the query carefully and identify if there is a time period mentioned in the query. You have to respond only with YES or NO depending on the query."
  # dateCheck = gpt_helper(query,dateCheckPrompt)
  # if dateCheck == "NO":
  #   return "Please provide a time period to get the response."


  lms_df = pd.read_csv("/Users/arihantbarjatya/Documents/finwin/lms_data.csv")
  credit_decisioning_df = pd.read_csv("/Users/arihantbarjatya/Documents/finwin/credit_decisioning_data.csv")
  lms_df["due_date"] = lms_df["due_date"].apply(dparser.parse)
  lms_df_filtered = lms_df[(lms_df['due_date']) > dparser.parse(start_dt)]
  lms_df_filtered = lms_df_filtered[(lms_df_filtered['due_date']) < dparser.parse(end_dt)]
  lms_df_filtered["defaulted_amount"] = lms_df_filtered["is_default"] * lms_df_filtered["loan_amount"]
  credit_decisioning_df = credit_decisioning_df.drop_duplicates()
  credit_decisioning_lms_df = pd.merge(lms_df_filtered, credit_decisioning_df, left_on = ["user_id"], right_on = ["user_id"], how = "left")
  credit_decisioning_lms_df = bin_df(credit_decisioning_lms_df, col_name, 5)
  col_group_name = col_name + "_groups"
  grouped_df = credit_decisioning_lms_df.groupby(col_group_name, dropna = False)[["defaulted_amount","loan_amount"]].sum().reset_index()
  grouped_df = credit_decisioning_lms_df.groupby(col_group_name, dropna = False).agg({'user_id' : 'count','defaulted_amount' : 'sum', 'loan_amount' : 'sum'}).reset_index()
  grouped_df["npa"] = grouped_df["defaulted_amount"]/ grouped_df["loan_amount"]
  grouped_df["fraction_of_users"] = grouped_df["user_id"]/grouped_df["user_id"].sum()
  if(col_name == 'bureau_score'):
    ntc_fraction = grouped_df[grouped_df[col_group_name].isnull()]['fraction_of_users'].iloc[0]
    non_ntc_fraction = 1-ntc_fraction
    ntc_npa = grouped_df[grouped_df[col_group_name].isnull()]['npa'].iloc[0]
    non_ntc_npa = grouped_df[~grouped_df[col_group_name].isnull()]['defaulted_amount'].sum()/grouped_df[~grouped_df[col_group_name].isnull()]['loan_amount'].sum()
    response = {'ntc_fraction' : round(ntc_fraction,2), 'non_ntc_fraction' : round(non_ntc_fraction,2), 'ntc_npa' : round(ntc_npa,2),'non_ntc_npa' : round(non_ntc_npa,2) }
    return grouped_df.to_string(), str(response)
  return grouped_df.to_string()


def calculate_top_features(external_data_lms_df, labels, num_fts = 10):
  features = external_data_lms_df
  gbm = ensemble.GradientBoostingRegressor()
  gbm.fit(features,labels)
  feature_importance = gbm.feature_importances_
  score_with_idx = []
  for idx,score in enumerate(feature_importance):
    score_with_idx.append((score,idx))
  sortedFeatures = sorted(score_with_idx)
  sortedFeatures.reverse()
  #===presently taking top 10 features only=====
  sortedFeatures = sortedFeatures[:num_fts]
  sortedIdx = [x[1] for x in sortedFeatures]
  df_with_top_fts = features.iloc[:,sortedIdx]
  top_fts = list(df_with_top_fts.columns)
  return top_fts

def evaluate_data_source():
  """
  This function is used to evaluate an external data source and test it's performance in predicting the behavior of a risk portfolio. 
  The external data source will have a number of input columns called features, and this function will calculate the most important or powerful features which can add most value in building my underwriting logics or risk model. 
  Here top_fts has a list of top 10 features.
  """
  #hardcoding external data source here @Arihant - please make the change for the user to i/p the data source
  external_data = pd.read_csv("/Users/arihantbarjatya/Documents/finwin/social-media_data.csv")
  external_data_lms_df = pd.merge(external_data, lms_df, left_on = ["user_id"], right_on = ["user_id"], how = "left")
  labels = external_data_lms_df["is_default"]
  external_data_lms_df.drop(["is_default"],axis=1,inplace=True)
  top_fts = calculate_feature_importance(external_data_lms_df,labels)
  risk_profiling_ft = "bureau_score"
  start_dt = loan_initiated_date.due_date.min
  end_dt = loan_initiated_date.due_date.max #for sake of simplicity start_dt and end_dt are taken to cover the full dataset. ideally the logic for date filtering should be more nuanced - to be added in future
  response = risk_profiling(start_dt, end_dt, risk_profiling_ft)
  #here both top_fts and response are to be processed wrt to a gpt_helper function. Here we might also need to output graphical trends
  return top_fts, response




if __name__ == "__main__":
  evaluate_data_source()
