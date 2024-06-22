import os, utils
import chainlit as cl
# from langchain_community.llms import huggingface_hub as HuggingFaceHub
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain


from getpass import getpass
HUGGINGFACEHUB_API_TOKEN=getpass()
os.environ["HUGGINGFACEHUB_API_TOKEN"]=HUGGINGFACEHUB_API_TOKEN

#models
#
# In machine learning, a pipeline refers to a series of data processing steps
# that are chained together in a specific sequence. Each step in the pipeline performs 
# a particular operation on the data, such as data preprocessing, feature extraction, model
# training, and model evaluation. Pipelines are commonly 
# used to streamline and automate
# the machine learning workflow, making it easier to manage and reproduce complex processes.

model_id = "gpt2-medium" #355pm
conv_model=HuggingFaceHub(huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
                         repo_id=model_id,
                         model_kwargs={"temperature":0.8,"max_new_tokens":200}
                         )
template=""" Explain

{query}"""

@cl.on_chat_start
def main():

    cl.user_session.set("conversation_history", [])  # Initialize the conversation history
    prompt=PromptTemplate(template=template, input_variables=['query'])
    conv_chain=LLMChain(llm=conv_model,prompt=prompt,verbose=True)
    cl.user_session.set("llm_chain",conv_chain)

    cl.user_session.set("llm_chain",conv_chain)

@cl.on_message
async def main(message:cl.Message):
    llm_chain=cl.user_session.get("llm_chain")
    conversation_history = cl.user_session.get("conversation_history", [])
    conversation_history.append((message.content, None))
    cl.user_session["conversation_history"] = conversation_history

    formatted_history = utils.format_conversation_history(conversation_history)
    res = await llm_chain.acall({"query": formatted_history + f"Human: {message.content}"}, callbacks=[cl.AsyncLangchainCallbackHandler()])

    conversation_history[-1] = (message.content, res["text"])
    cl.user_session["conversation_history"] = conversation
    # do postprocessing after receiving response
    # res is dict and all response is stored under text key
    await cl.Message(content=res["text"]).send()
    print(message)


