import nltk
import numpy as np
import torch
from lexrank import degree_centrality_scores
from sentence_transformers import SentenceTransformer, util

TRANSFORMERS_CACHE="/f/C/cache/huggingface/hub"

# model = SentenceTransformer('all-MiniLM-L6-v2')

model = SentenceTransformer('all-mpnet-base-v2')

# Our input document we want to summarize
# As example, we take the first section from Wikipedia
text = """
 Did you know that you could be masking and compensating without even realizing you're doing it? So compensation is basically the techniques that we employ so that we don't present as autistic. So you could have an autistic neurology, but you employ certain techniques to present as holistic in social situations, which is how some autistic people can present entirely holistic, even though they are autistic. So there are shallow compensation skills and shallow compensation would look like me saying, okay, if somebody makes a joke in any social dynamic, I will laugh at it. That is a across-the-board rule that I'm making to look like I fit in and look like I have a similar sense of humor. But there's also something called deep compensation and this is the important part because deep compensation can become second nature to where you don't realize you're doing it and it can make you present like an holistic without even realizing it. This weekend I took the reading the mind in the eyes test on embraceautism.com and this is a assessment for a theory of mind and I thought it would be easy for me because I'm a therapist and I pick up on emotions really easily with people, but it basically just shows pictures of the eyes and you're supposed to see if you know what emotion they're feeling and this was surprisingly difficult because apparently I use the whole face to tell what emotion someone's feeling, not just the eyes like holistics do or rather like they can. So I took this assessment and it took me a really long time and I felt like I was guessing on all of them except maybe two or three, but I got an almost holistic score and the reason that I did this is because I was able to employ what I know about facial expressions. I didn't intuitively know the emotion, but I was able to analyze it. I was able to look at where the cheeks lifted, where the eyebrows furrowed, what would that insinuate about how that person is feeling and then I implied that in order to guess on what I thought the answer was. So while maybe I was able to do the same thing as an holistic, I did it in a completely different way. But because you're not inside the head of holistics, you don't know how they do it. You don't know how they socialize or you know, what information they're using to make these decisions. So all you really know is that you're coming up with the same results as they are, which might make you feel like you're probably holistic too. And while researchers used to think that if an autistic person no longer presented with the behavioral struggles and they were able to mask and compensate that that would mean that they're no longer autistic. But what we now know is they still have autistic neurology in the same underpinning cognitive structure. It's just the outward presentation that has changed. It does not change the fact that they are indeed autistic.
 """

len(text)

f=len(text.split())
print ("The number of words in the given text is : " +  str(f))


from string import punctuation

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp= spacy.load("en_core_web_sm")
doc=nlp(text)
tokens=[token.text for token in doc]
print(tokens)

punctuation=punctuation+'n'


word_freq={}
stop_words= list(STOP_WORDS)

# A loop has been run over the doc to get those words that are not in the list of STOP_WORDS and also not in the list of punctuations, and then the words were added to the word_freq dictionary and the number of times they appear in doc has been added as a value in the dictionary.
for word in doc:
   if word.text.lower() not in stop_words:
     if word.text.lower() not in punctuation:
       if word.text not in word_freq.keys():
         word_freq[word.text]= 1
       else:
         word_freq[word.text]+= 1 
print(word_freq)

# The maximum no of times a word appear has been figured out stored in variable max_freq.
x=(word_freq.values())
a=list(x)
a.sort()
max_freq=a[-1]
max_freq

# All the score of the words in word_freq dictionary has been normalized by dividing each value in the dictionary by max_freq and to do this a loop has been run on word_freq dictionary and all the values were normalized.
# Sentence Tokenization

for word in word_freq.keys():
   word_freq[word]=word_freq[word]/max_freq
print(word_freq)

# Sentences in doc objects have been segmented by using the list comprehension method and kept in variable sent_tokens.
sent_score={}
sent_tokens=[sent for sent in doc.sents]
print(sent_tokens)


# A score of each individual sentence has been found out based on the word_freq counter. An empty dictionary sent_score has been created which will hold each sentence as a key and its value as a score. A loop was iterated on each individual sentence and it was checked the words in those sentences if appear in word_freq dictionary and then based on the score of a word in word_freq dictionary sent_score has been determined.

for sent in sent_tokens:
    for word in sent:
        if word.text.lower() in word_freq.keys():
            if sent not in sent_score.keys():
                sent_score[sent]=word_freq[word.text.lower()]
            else:
                sent_score[sent]+= word_freq[word.text.lower()] 
print(sent_score)


# A priority queue is commonly represented using the data structure ‘heap.’ The heapq module in the Python standard library can be used to carry out this implementation. The functions of the heapq module serve the goal of choosing the best element. In Python, the heap data structure has the feature of always popping the smallest heap member (min-heap). The heap structure is preserved whenever data pieces are popped or pushed.
# From heapq module nlargest library was imported. from the total sent_score, 30% has been evaluated which comes to 13, which means a maximum of 13 sentences can be extracted which contains all important information.
from heapq import nlargest

len(sent_score) *0.3


print('\nsummary\n')
summary=nlargest(n=13,iterable=sent_score,key=sent_score.get) 
print(summary)

# List comprehension was applied to get the final summarized text.

final_summary=[word.text for word in summary]
final_summary

import re

# Empty list f1 was created and a loop was run on the final extracted text, then regex operation was done to remove ‘n’ from all text and appended to list f1.

f1=[]
for sub in final_summary:
    f1.append(re.sub('n','',sub))
f1
# The list of final summarized text was converted to string using the join() function and kept in variable f2.
f2=" ".join(f1)
f2

f3=len(f2.split())
print ("The number of words in final summary is : " +  str(f3))

"""
import spacy
from spacy.lang.en.examples import sentences 

nlp = spacy.load("en_core_web_sm")
doc = nlp(sentences[0])
print(doc.text)
for token in doc:
    print(token.text, token.pos_, token.dep_)
"""


"""
#Split the document into sentences
sentences = nltk.sent_tokenize(document)
print("Num sentences:", len(sentences))

#Compute the sentence embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)

#Compute the pair-wise cosine similarities
# cos_scores = torch.nn.functional.cosine_similarity(embeddings, embeddings).cpu().numpy()
cos_scores = torch.mm(embeddings, embeddings.t()) / torch.mm(torch.norm(embeddings, p=2, dim=1, keepdim=True), torch.norm(embeddings, p=2, dim=1, keepdim=True).t())

#Compute the centrality for each sentence
centrality_scores = degree_centrality_scores(cos_scores, threshold=None)

#We argsort so that the first element is the sentence with the highest score
most_central_sentence_indices = torch.argsort(-centrality_scores)


#Print the 5 sentences with the highest scores
print("\n\nSummary:")
for idx in most_central_sentence_indices[0:5]:
    print(sentences[idx].strip())

"""