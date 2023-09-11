from pydantic import BaseModel
from typing import List
from data_model import CompanySchema
import openai
import json
from langchain.prompts import PromptTemplate
from langchain.prompts import StringPromptTemplate
from langchain.llms import OpenAI




class DealGenerator():

	def __init__(self):
		self.outputKeys = ["description"]
		# self.company : str = promptData.named
		# self.industry : str = promptData.industry
		# self.bPlan : str = promptData.bPlan
		# self.targetIndustry : str = promptData.targetIndustry
		# self.deals : int = promptData.dealsRequired

	def create_prompt(self,data,dealsRequired):
		companyOne = str(json.loads(data[0].json()))
		companyTwo = str(json.loads(data[1].json()))
		n = dealsRequired
		
		
		INSTRUCTION_PROMPT = f"""
		Instruction: \n
		Act as a Business Development Manager. You will be provided a number(n) and two json object each containing description of a company. Analyse the company details
		and come up with n realistic business partnership deals between the two companies. Each Business Deal should be in json format with keys:\n
		{str(self.outputKeys)}\n
		The final output should be in this json format , with key value pairs in this format:
		
		"response":"[Business Deal 1, Business Deal 2, ...... Business Deal n]"
		"numberOfDeals":"n"

		The output should just be this json object.
		Dont make anything up, If you dont have any response just return Sorry, I cant help you.
		"""

		# EXAMPLES_PROMPT = f"""
		# Take inspirations from the following examples\n
		# Examples:\n
		# {self.examples[0]} \n
		# {self.examples[1]} \n
		# """

		INPUT_PROMPT = f"""
		Now generate response for the following input.\n
		Company #1 Details:\n
		{companyOne}\n
		Company #2 Details:\n
		{companyTwo}\n
		Required Number of Business Deals:\n
		{n}

		Response:
		"""

		template = INSTRUCTION_PROMPT + INPUT_PROMPT
		#print(template)
		#finalPrompt = StringPromptTemplate(template)
		#finalPrompt.format(outputKeys=str(self.outputKeys),companyOne=companyOne,companyTwo=companyTwo,n=n)
		#print(finalPrompt)
		#print(type(template))
		return template


	# def validate_response(response,ogNumberOfDeals):
	# 	dealList = response["response"]
	# 	numberOfDeals = response["numberOfDeals"]
	# 	if !assert(numberOfDeals,ogNumberOfDeals):
	# 		#add logic to push wrong response and demand for correct output from openAI
	# 		pass

	# 	if !assert(len(dealList),ogNumberOfDeals):
	# 		#add logic to push wrong response and demand for correct output from openAI
	# 		pass
	# 	return response


	# add logic for validation that response is json
	def generate_deals(self,data,numberOfDeals):
		openai.api_key="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"
		#llm = OpenAI()
		finalPrompt = self.create_prompt(data,numberOfDeals)
		#print(finalPrompt)
		#print(finalPrompt)
		model_id = "text-davinci-003"
		responseRaw = openai.Completion.create(engine=model_id,prompt=finalPrompt,temperature=0.6,max_tokens=3000)
		response = json.loads(responseRaw["choices"][0]["text"])
		return response








