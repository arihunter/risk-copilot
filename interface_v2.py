import streamlit as st 
from agent import OpenAIAgent
import openai 
from langchain.agents.tools import Tool
from langchain.chat_models import ChatOpenAI
from llama_index.tools.function_tool import FunctionTool
import pandas as pd
import dateutil.parser as dparser
import os
from typing import List
from google.oauth2 import service_account
import gspread
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier

openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_resource(show_spinner=False)
def gsheets_connection():
  credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"],scopes=["https://www.googleapis.com/auth/spreadsheets","https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"])
  gc = gspread.authorize(credentials)
  sheet_url = st.secrets["private_gsheets_url"]
  sheet = gc.open_by_url(sheet_url)
  return sheet
feedbackSheet = gsheets_connection()

st.set_page_config(page_title="Stealth")
col1,col2,col3 = st.columns([1,1,1])
col2.title("Stealth")
st.divider()


#initialisation of session state variables
if "feedback" not in st.session_state.keys():
  st.session_state.feedback = False
if "prompt" not in st.session_state.keys():
  st.session_state.feedback = None
if "response" not in st.session_state.keys():
  st.session_state.feedback = None

dataset_keys = ["lms","credit-decisioning","collection","social-media"]
for key in dataset_keys:
  if key not in st.session_state :
    st.session_state[key] = False



# agent tools definitions

def gpt_helper(query:str,context:str) -> str:
  SYSTEM_PROMPT = f"{context}"
  model_id = "gpt-4"
  response = openai.ChatCompletion.create(
    model=model_id,
    messages=[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":query}
    ],
    max_tokens=3000
  )
  return response["choices"][0]["message"]["content"]


def calculate_risk_metrics(start_dt:str,end_dt:str) -> float:
  """
  This function is used to calculate the risk performance metrics of a lending portfolio, for a given time period. Risk performance metrics include the following:

  Disbursal amount or the loan amount which is defined as the loan amount disbursed.
  Defaulted amount which is defined as the loan amount which is defaulted or the amount which is due.
  Number of loans disbursed which is defined as the number of people approved for a loan.
  Average ticket size which is defined as the average value of the loan amount disbursed. It is calculated as a ratio of the disbursal amount to number of loans disbursed.
  NPA(non-performing asset) which is defined as the percentage of disbursed amount which is due or defaulted.

  """
  dataset_idx=[0]
  for idx in dataset_idx:
    if st.session_state[dataset_keys[idx]] == False:
      return "Sufficient data not available !, please provide all the required data."


  # input health check
  dateCheckPrompt = """
  I will provide you with a user query, you have to analyse the query carefully and identify if there is a complete time period mentioned in the query. 
  If you're able to identify all date elements from the user query you have to respond with YES else NO.
  Here are some examples for you.

  Example 1:
  Query: Calculate defaulted amount for the third financial quarter of 2021.
  Response: YES

  Example 2:
  Query : could you provide me the NPA calculation of the year 2022. I need it for each quarter individually
  Response: YES
  
  Example 3:
  Query : could you provide me the NPA calculation of the first quarter
  Response: NO.
  """
  
  dateCheck = gpt_helper(prompt,dateCheckPrompt)
  if dateCheck == "NO":
    return "There was insufficent date information to do the calculations. Ask the user to give complete date information in the query. Do not make up any random metrics by yourself."

  lms_df = pd.read_csv("lms_data.csv")
  lms_df["due_date"] = lms_df["due_date"].apply(dparser.parse)
  lms_df_filtered = lms_df[(lms_df['due_date']) > dparser.parse(start_dt)]
  lms_df_filtered = lms_df_filtered[(lms_df_filtered['due_date']) < dparser.parse(end_dt)]
  lms_df_filtered["defaulted_amount"] = lms_df_filtered["is_default"] * lms_df_filtered["loan_amount"]
  defaulted_amount = lms_df_filtered['defaulted_amount'].sum()
  disbursal_amount = lms_df_filtered['loan_amount'].sum()
  number_of_loans_disbursed = lms_df_filtered['loan_amount'].count()
  average_ticket_siZe = disbursal_amount/number_of_loans_disbursed
  portfolio_npa = defaulted_amount / disbursal_amount
  context = {'defaulted_amount' : round(defaulted_amount,2), 'disbursal_amount' : round(disbursal_amount,2), 'number_of_loans_disbursed' : round(number_of_loans_disbursed,2), 'portfolio_npa' : round(portfolio_npa,2)}
  
  #response quality check
  example_context1 = {'defaulted_amount': 189256.01, 'disbursal_amount': 1020599.01, 'number_of_loans_disbursed': 4, 'portfolio_npa': 0.19}
  example_context2 = {'defaulted_amount': 1687413.0, 'disbursal_amount': 9991314.34, 'number_of_loans_disbursed': 45, 'portfolio_npa': 0.17}
  responsePrompt = f"""
  I need you to answer the users query using the given context. 
  The response to the query is certain to be in the context.
  Go carefully through the query and context and just return the answer, nothing else.
  Dont make anything up. Dont do any calculations on your end. Do not assume any denomination for the requested metrics in the query. 
  Here are some examples for you.

  Example 1:
  Context : {example_context1}
  Query: Calculate defaulted amount for the third financial quarter of 2021.
  Response: The defaulted amount for the third financial quarter of 2021 is 189256.01.

  Example 2:
  Context : {example_context2}
  Query : What is the NPA metric for the time period "01/04/2021" to "31/08/2021"
  Response: The NPA metric for the time period "01/04/2021" to "31/08/2021" is 0.17.

  Now using the given context answer the query.
  Context:
  {str(context)}
  """
  print(f"The values calculated are {context}")
  response = gpt_helper(prompt,responsePrompt)
  print(f"Gpt helper response for risk metric function output formatting {response}")
  return response



def bin_df(df:pd.DataFrame,col_name:str,number_of_bins:int=5) -> pd.DataFrame:
  col_group_name = col_name + "_groups"
  df[col_group_name] = pd.qcut(df[col_name], number_of_bins)
  return df


def risk_profiling(start_dt:str,end_dt:str,col_name:str="bureau_score") -> str:
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
  dateCheckPrompt = "I will provide you with a user query , you have to analyse the query carefully and identify if there is a time period mentioned in the query. You have to respond only with YES or NO depending on the query."
  dateCheck = gpt_helper(prompt,dateCheckPrompt)
  if dateCheck == "NO":
    return "There was insufficent date information to do the calculations. Ask the user to give complete date information in the query. Do not make up any random metrics by yourself."


  lms_df = pd.read_csv("lms_data.csv")
  credit_decisioning_df = pd.read_csv("credit-decisioning_data.csv")
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
    response = {'ntc_fraction' : ntc_fraction, 'non_ntc_fraction' : non_ntc_fraction, 'ntc_npa' : ntc_npa,'non_ntc_npa' : non_ntc_npa, 'total_users' : grouped_df["user_id"].sum()}
    return grouped_df.to_string(), str(response)
  return grouped_df.to_string()


# def calculate_top_features(external_data_lms_df, labels, num_fts = 10):
#   features = external_data_lms_df
#   gbm = ensemble.GradientBoostingRegressor()
#   gbm.fit(features,labels)
#   feature_importance = gbm.feature_importances_
#   score_with_idx = []
#   for idx,score in enumerate(feature_importance):
#     score_with_idx.append((score,idx))
#   sortedFeatures = sorted(score_with_idx)
#   sortedFeatures.reverse()
#   #===presently taking top 10 features only=====
#   sortedFeatures = sortedFeatures[:num_fts]
#   sortedIdx = [x[1] for x in sortedFeatures]
#   df_with_top_fts = features.iloc[:,sortedIdx]
#   top_fts = list(df_with_top_fts.columns)
#   return top_fts

# def evaluate_data_source():
#   """
#   This function is used to evaluate an external data source and test it's performance in predicting the behavior of a risk portfolio. 
#   The external data source will have a number of input columns called features, and this function will calculate the most important or powerful features which can add most value in building my underwriting logics or risk model. 
#   Here top_fts has a list of top 10 features.
#   """
#   #hardcoding external data source here @Arihant - please make the change for the user to i/p the data source
#   external_data = pd.read_csv("/Users/arihantbarjatya/Documents/finwin/social-media_data.csv")
#   lms_df = pd.read_csv("/Users/arihantbarjatya/Documents/finwin/lms_data.csv")
#   external_data_lms_df = pd.merge(external_data, lms_df, left_on = ["user_id"], right_on = ["user_id"], how = "left")
#   labels = external_data_lms_df["is_default"]
#   external_data_lms_df.drop(["is_default"],axis=1,inplace=True)
#   top_fts = calculate_feature_importance(external_data_lms_df,labels)
#   risk_profiling_ft = "bureau_score"
#   start_dt = loan_initiated_date.due_date.min
#   end_dt = loan_initiated_date.due_date.max #for sake of simplicity start_dt and end_dt are taken to cover the full dataset. ideally the logic for date filtering should be more nuanced - to be added in future
#   response = risk_profiling(start_dt, end_dt, risk_profiling_ft)
#   #here both top_fts and response are to be processed wrt to a gpt_helper function. Here we might also need to output graphical trends
#   return top_fts, response



#Agent creation
RiskProfileTool = FunctionTool.from_defaults(fn=risk_profiling)
RiskMetricsTool = FunctionTool.from_defaults(fn=calculate_risk_metrics)
#EvaluateDataTool = FunctionTool.from_defaults(fn=evaluate_data_source)



#Tools Llamaindex
llamaTools = [RiskMetricsTool,RiskProfileTool]
agentLlama = OpenAIAgent.from_tools(llamaTools)



#UI starts

def UploadedFileCallback(displayText:str):
  print(displayText)

with st.sidebar:
  DatasetOption = st.selectbox("Choose the Dataset",("LMS","Credit-Decisioning","Collection","Social-Media"))
  
  #Data Upload
  if st.session_state[str(DatasetOption.lower())] == True:
    st.warning("Uploading now will update the dataset")
    displayText = "Updating the " + DatasetOption + "dataset"
    UploadedFile = st.file_uploader("Update the file here",type="csv",on_change=UploadedFileCallback,args=[displayText])
    if UploadedFile is not None:
      dataset_path = str(DatasetOption.lower()) + "_data.csv"
      string_path = str(DatasetOption.lower()) + "_data.txt"
      with st.spinner("Updating"):
        pd.read_csv(UploadedFile).to_csv(dataset_path)
  else:
    displayText = "Uploading the " + DatasetOption + "dataset"
    UploadedFile = st.file_uploader("Upload the file here",type="csv",on_change=UploadedFileCallback,args=[displayText])
    if UploadedFile is not None:
      st.session_state[str(DatasetOption.lower())] = True
      dataset_path = str(DatasetOption.lower()) + "_data.csv"
      string_path = str(DatasetOption.lower()) + "_data.txt"
      with st.spinner("Uploading"):
        pd.read_csv(UploadedFile).to_csv(dataset_path)


userEmail = st.experimental_user.email

def find_last_filled_row(worksheet):
  return len(worksheet.get_all_values()) + 1

def insert_data_into_sheet(dataframe):
  worksheet = feedbackSheet.get_worksheet(0)  # Replace 0 with the index of your desired worksheet
  values = dataframe.values.tolist()
  last_filled_row = find_last_filled_row(worksheet)
  worksheet.insert_rows(values, last_filled_row)

#Chat Interface
def ResponseCallback(prompt:str,response:str,_type:str):
  st.session_state.feedback = True
  feedbackDict = {"Prompt":[prompt],"Response":[response],"Feedback":[_type],"UserId":[userEmail]}
  feedbackDf = pd.DataFrame.from_dict(feedbackDict)
  insert_data_into_sheet(feedbackDf)

def ChatInputCallback():
  st.session_state.feedback = False
 
@st.cache_data(show_spinner=False)
def get_response(query:str) -> str:
	response = agentLlama.chat(query)
	return response

#Input
global prompt
if prompt := st.text_input("Enter Here",on_change=ChatInputCallback):
	st.session_state.prompt = prompt
	with st.spinner("Thinking"):
		response = get_response(prompt)
		placeholder = st.empty()
		placeholder.write(f'<i>{response}</i>',unsafe_allow_html=True)
		relevantCol1,relevantCol2,relevantCol3 = placeholder.columns([0.8,0.1,0.1])
		with relevantCol2:
			if st.session_state.feedback == False:
				placeholder.button(":thumbsup:",on_click=ResponseCallback,args=([str(prompt),str(response),"POSITIVE"]),disabled=False)
		with relevantCol3:
			if st.session_state.feedback == False:
				placeholder.button(":thumbsdown:",on_click=ResponseCallback,args=([str(prompt),str(response),"NEGATIVE"]),disabled=False)	




