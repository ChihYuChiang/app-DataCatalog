import pandas as pd
import json
import re
import nltk
from gensim.models import TfidfModel
from gensim.corpora import Dictionary




#--Loading data back
#Main df
df = pd.read_csv('./data/df_joined.csv', index_col=False)
df.head()

#Category tag and description
with open('./data/dict_category.json', 'r') as f:
    cat = json.load(f)
    print(cat)

#Combine title, overview, and description columns
ats = df['title'].str.cat(df['overview'], sep='. ').str.cat(df['description'], sep='. ')


#--POS tagging and filter
#Tokenize and tag
ats_token = [nltk.word_tokenize(at) for at in ats]
ats_tag = [nltk.pos_tag(at) for at in ats_token]

#Preserve only nouns and [a-zA-Z0-9\-]
def filterTag(targetTags, at):
    at_filteredTag = ((word, tag) for (word, tag) in at if tag in targetTags)
    return at_filteredTag

def filterWord(pattern, at):
    at_filteredWord = ((word, tag) for (word, tag) in at if re.search(pattern, word))
    return at_filteredWord

ats_filtered = [list(filterWord('^[a-zA-Z0-9\-]+$', filterTag(['NN', 'NNP', 'NNS'], at))) for at in ats_tag]

#Combine POS and word
def appendTag(at):
    at_tagAppended = (word + '_' + tag for (word, tag) in at)
    return at_tagAppended

ats_tagAppended = [list(appendTag(at)) for at in ats_filtered]


#--Tf-idf
#Create dictionary
dct = Dictionary(ats_tagAppended)
dct[0]

#Convert to bag of word format
ats_bow = [dct.doc2bow(at) for at in ats_tagAppended]

#Model
model = TfidfModel(ats_bow)

#Apply model to an article
ats_tfIdf = [model[at] for at in ats_bow]

#Filter by tf-idf and convert back to words
def filterTfIdf(at, threshold, dct):
    at_tfIdfFiltered = ((dct[idx], score) for (idx, score) in at if score >= threshold)
    return at_tfIdfFiltered

ats_tfIdfFiltered = [list(filterTfIdf(at, 0.2, dct)) for at in ats_tfIdf]
