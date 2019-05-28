import pandas as pd
import json
import nltk
from gensim.models import TfidfModel
from gensim.corpora import Dictionary


#--Loading data back
df = pd.read_csv('./data/df_joined.csv', index_col=False)
df.head()
with open('./data/dict_category.json', 'r') as f:
    cat = json.load(f)
    print(cat)


#--Cleaning
[i for i in df['description']]


#Tf-idf
ttt = [nltk.word_tokenize(txt) for txt in df['description'][0:5]]
dct = Dictionary(ttt)  # fit dictionary
corpus = [dct.doc2bow(at) for at in ttt]  # convert corpus to BoW format
model = TfidfModel(corpus)  # fit model
vector = model[corpus[0]]  # apply model to the first corpus document
dct[71]

#----
df['description'][2]
lTokens = nltk.word_tokenize(df['description'][0])
lTags = nltk.pos_tag(lTokens)

def findtags(tagged_text):
    cfd = nltk.ConditionalFreqDist((tag, word) for (word, tag) in tagged_text)
    for tag in cfd.conditions():
        print(tag, dict(cfd[tag]))
    return dict((tag, cfd[tag].keys()) for tag in cfd.conditions())

lTagDict = findtags(lTags)

#Identify nouns, use tf-idf to weight
#Tokenize based on PN


#--Bond