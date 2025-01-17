from openbb_terminal.sdk import openbb

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0)

print("Hello World")

prompt = """
Is the predominant sentiment in the following statement positive, negative, or neutral?
---------
Statement: {statement}
---------
Respond with one word in lowercase: positive, negative, or neutral.
Sentiment:
"""

chain = LLMChain.from_string(
    llm=llm,
    template=prompt
)

SPDR = openbb.news(term="SPDR")
SPDR["Sentiment"] = SPDR.Description.apply(chain.run)
SPDR[["Description", "Sentiment"]]
print(SPDR)
print("completed")