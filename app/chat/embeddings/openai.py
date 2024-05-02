from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

embeddings = OpenAIEmbeddings()
