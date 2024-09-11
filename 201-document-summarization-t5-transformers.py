from transformers import pipeline
import os

TRANSFORMERS_CACHE="/f/C/cache/huggingface/hub"

def read_file(file_name: str) -> str:
    file = os.path.join(r'F:\R\Repositories\snippets\youtube_transcriber\transcript', file_name)
    with open(file, encoding="utf8") as file:
        return file.read()

summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="pt")
text = read_file('36a883acde3bb5cfb4e70df225d66532.txt')
summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
print("First summary thing\n")
print(summary)

from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelWithLMHead.from_pretrained("t5-base")

print("Summarize num two")
def summarize(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0])
print(summarize(text))