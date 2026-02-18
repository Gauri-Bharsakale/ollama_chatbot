# streamlit + langchain + ollama (gemma2:2b model)

import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# step1 - create prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond clearly to the question asked."),
        ("user", "Question: {question}")
    ]
)

# step2 - streamlit app ui
st.title("LangChain Demo with Gemma Model (Ollama)")

input_txt = st.text_input("What question do you have in your mind?")

# step3 - load ollama model
llm = Ollama(model="gemma2:2b")

# output parser
output_parser = StrOutputParser()

# pipeline
chain = prompt | llm | output_parser

# step4 - run model when user enters question
if input_txt:
    response = chain.invoke({"question": input_txt})
    st.write(response)
