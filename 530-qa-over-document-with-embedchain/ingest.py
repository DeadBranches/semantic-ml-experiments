import os

os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

from embedchain import App

rights_bot = App()
rights_bot.add_local("pdf_file", "../sources/policy_on_abelism.pdf")
print("done")
