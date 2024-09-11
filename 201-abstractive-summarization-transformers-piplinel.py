import transformers

f = open("sources-documents/wikipedia-new-york-city-excerpt.txt", "r", encoding="latin1")

to_tokenize = f.readlines()

summarizer = pipeline("summarization")

summarized = summarizer(to_tokenize, min_length=75, max_length=300)
print(summarized)

summ=' '.join([str(i) for i in summarized])

summ