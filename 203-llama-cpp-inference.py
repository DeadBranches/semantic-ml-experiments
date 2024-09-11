from llama_cpp import Llama

llm = Llama(model_path="F:/M/Wizard-Vicuna-30B-Uncensored.ggmlv3.q5_1.bin")
output = llm(
    "USER: Name the planets in the solar system? ASSISTANT:",
    max_tokens=32,
    stop=["Q:", "\n"],
    echo=True,
)
print(output)
