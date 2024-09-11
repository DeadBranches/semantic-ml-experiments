import os

os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxx"

from embedchain import App


rights_bot = App()
question = "Is it discrimination to be mean to another person?"
print(f"Query: {question}")
print(rights_bot.query(question))
