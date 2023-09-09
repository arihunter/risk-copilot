import sys
#sys.path.append('../')
import pandas as pd
import dateutil.parser as dparser
import sys
from llama_index.tools.function_tool import FunctionTool
from llama_index.agent import OpenAIAgent
from langchain.agents.tools import Tool
from langchain.chat_models import ChatOpenAI
import streamlit as st
import openai
import os
from typing import List
import csv
# from trubrics.integrations.streamlit import FeedbackCollector



openai.api_key = "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"
os.environ["OPENAI_API_KEY"] = "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"

#Frontend Initialisation

st.set_page_config(page_title="Stealth")
col1,col2,col3 = st.columns([1,1,1])
col2.title("Risk Copilot")
st.divider()


if "messages" not in st.session_state.keys():
  st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

if "thumbsdown" not in st.session_state.keys():
  st.session_state.thumbsdown = False

if "thumbsup" not in st.session_state.keys():
  st.session_state.thumbsup = False

dataset_keys = ["lms","credit-decisioning","collection","social-media"]
for key in dataset_keys:
  if key not in st.session_state :
    st.session_state[key] = False

# @st.cache_data(show_spinner=False)
# def feedback_collector():
#   collector = FeedbackCollector(project="default",email="arihantbarjatya2@gmail.com",password="stealth")
#   return collector

# FeedbackCollector = feedback_collector()

# stringFile = {}
# for key in dataset_keys:
#   if key not in stringFile:
#     stringFile[key] = None

# csvFile = {}
# for key in dataset_keys:
#   if key not in csvFile:
#     csvFile[key] = None


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
  print(response)
  return response["choices"][0]["message"]["content"]



#TODO
#write functions on the basis of query?


#Tools Definations
#TODO - Make descriptions better.

#start_dt:str, end_dt:str
#terminating function 

#asking for input in the UI is difficult , due to agent dying very early.


def calculate_risk_metrics(start_dt:str,end_dt:str,query:str) -> float:
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


  #query check
  # dateCheckPrompt = "I will provide you with a user query , you have to analyse the query carefully and identify if there is a time period mentioned in the query. You have to respond only with YES or NO depending on the query."
  # dateCheck = gpt_helper(query,dateCheckPrompt)
  # if dateCheck == "NO":
  #   return "Please provide a time period to get the response."


  lms_df = pd.read_csv("/Users/arihantbarjatya/Documents/finwin/local_storage/lms_data.csv")
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
  
  example_context1 = {'defaulted_amount': 189256.01, 'disbursal_amount': 1020599.01, 'number_of_loans_disbursed': 4, 'portfolio_npa': 0.19}
  example_context2 = {'defaulted_amount': 1687413.0, 'disbursal_amount': 9991314.34, 'number_of_loans_disbursed': 45, 'portfolio_npa': 0.17}
  responsePrompt = f"""
  I need you to answer the users query using the given context. 
  The response to the query is certain to be in the context.
  Go carefully through the query and context and just return the answer , nothing else.
  Dont make anything up. Dont do any calculations on your end.
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
  response = gpt_helper(query,responsePrompt)
  print(response)
  return response



#intermediate function
def bin_df(df:pd.DataFrame,col_name:str,number_of_bins:int=5) -> pd.DataFrame:
  col_group_name = col_name + "_groups"
  df[col_group_name] = pd.qcut(df[col_name], number_of_bins)
  return df


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


  dateCheckPrompt = "I will provide you with a user query , you have to analyse the query carefully and identify if there is a time period mentioned in the query. You have to respond only with YES or NO depending on the query."
  dateCheck = gpt_helper(query,dateCheckPrompt)
  if dateCheck == "NO":
    return "Please provide a time period to get the response."


  lms_df = pd.read_csv("/Users/arihantbarjatya/Documents/finwin/local_storage/lms_data.csv")
  credit_decisioning_df = pd.read_csv("/Users/arihantbarjatya/Documents/finwin/local_storage/credit_decisioning_data.csv")
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







#Llama Index Tool Creation
#NpaTool = FunctionTool.from_defaults(fn=calculate_npa)
RiskProfileTool = FunctionTool.from_defaults(fn=risk_profiling)
RiskMetricsTool = FunctionTool.from_defaults(fn=calculate_risk_metrics)



#Tools Llamaindex
llamaTools = [RiskMetricsTool,RiskProfileTool]
agentLlama = OpenAIAgent.from_tools(llamaTools)


#Langchain Tool Creation
# tools = [
#     Tool(
#         name = "NPA Calculator",
#         func=calculate_npa,
#         description="useful to calculate NPA metric of a lending portfolio , for a given time period."
#     ),
#     Tool(
#         name="Risk Profiler",
#         func=risk_profiling,
#         description="useful to do risk profiling of borrowers, for a given time period."
#     ),
# ]


# #Agent Langchain
# model = ChatOpenAI(temperature=0)
# planner = load_chat_planner(model)
# executor = load_agent_executor(model, tools, verbose=True)
# agentLangchain = PlanAndExecute(planner=planner, executor=executor, verbose=True)




#Frontend Starts




# Store LLM generated responses


# Display or clear chat messages
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.write(message["content"])

def clear_chat_history():
  st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

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
      dataset_path = "/Users/arihantbarjatya/Documents/finwin/local_storage/" + str(DatasetOption.lower()) + "_data.csv"
      string_path = "/Users/arihantbarjatya/Documents/finwin/local_storage/" + str(DatasetOption.lower()) + "_data.txt"
      with st.spinner("Updating"):
        pd.read_csv(UploadedFile).to_csv(dataset_path)
  else:
    displayText = "Uploading the " + DatasetOption + "dataset"
    UploadedFile = st.file_uploader("Upload the file here",type="csv",on_change=UploadedFileCallback,args=[displayText])
    if UploadedFile is not None:
      st.session_state[str(DatasetOption.lower())] = True
      dataset_path = "/Users/arihantbarjatya/Documents/finwin/local_storage/" + str(DatasetOption.lower()) + "_data.csv"
      string_path = "/Users/arihantbarjatya/Documents/finwin/local_storage/" + str(DatasetOption.lower()) + "_data.txt"
      with st.spinner("Uploading"):
        pd.read_csv(UploadedFile).to_csv(dataset_path)
  
  st.button('Clear Chat History', on_click=clear_chat_history)

userEmail = st.experimental_user.email



#@st.cache_resource(show_spinner=False)
def ResponseCallback(_prompt:str,_response:str,_type:str):
  # field_names = ['PROMPT', 'RESPONSE', 'userID']
  file = "BACKUP_" + _type + "_FILE"
  # dict_ = {'Prompt': _prompt, 'RESPONSE': _response, 'userID': userEmail}
  if file=="BACKUP_POSITIVE_FILE":
    with open(BACKUP_POSITIVE_FILE, 'a') as f_object:
      f_object.write(f'\n{str(_prompt)},{str(_response)},{userEmail}')
      # dictwriter_object = csv.DictWriter(f_object, fieldnames=field_names)
      # dictwriter_object.writerow(dict_)
      # print(f"written the new data to {file}")
      f_object.close()
  elif file=="BACKUP_NEGATIVE_FILE":
    with open(BACKUP_NEGATIVE_FILE, 'a') as f_object:
      f_object.write(f'\n{str(_prompt)},{str(_response)},{userEmail}')
      # dictwriter_object = csv.DictWriter(f_object, fieldnames=field_names)
      # dictwriter_object.writerow(dict_)
      # print(f"written the new data to {file}")
      f_object.close()

def ChatInputCallback():
  st.session_state.thumbsup = False
  st.session_state.thumbsdown = False


#Chat Interface
if prompt := st.chat_input("Type Here",on_submit=ChatInputCallback):
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.write(prompt)


if st.session_state.messages[-1]["role"] != "assistant":
  with st.chat_message("assistant"):
    with st.spinner("Thinking"):
      relevantCol1,relevantCol2,relevantCol3 = st.columns([0.8,0.1,0.1])
      response = agentLlama.chat(prompt)
      #response = agentLangchain.run(prompt)
      with relevantCol1:
        placeholder = st.empty()
        placeholder.markdown(str(response))



  message = {"role": "assistant", "content": str(response)}
  st.session_state.messages.append(message)

