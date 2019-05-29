import pandas as pd
import json
import re
import nltk
from gensim.models import TfidfModel
from gensim.corpora import Dictionary




#--Loading data back
#Main df
df = pd.read_csv('./data/df_joined.csv', index_col=False)
df.fillna('', inplace=True)
df.head()

#Category tag and description
with open('./data/dict_category.json', 'r') as f:
    cat = json.load(f)

#Combine title, overview, and description columns
ats = df['title'].str.cat(df['overview'], sep='. ').str.cat(df['description'], sep='. ')


#--POS tagging and filter
#Tokenize and tag
ats_token = [nltk.word_tokenize(at) for at in ats]
ats_tag = [nltk.pos_tag(at) for at in ats_token]

#Combine NNP
def combineNNP(at):
    at.append(('$', 'END'))
    at_combinedNNP = []
    word_prev, tag_prev = '', ''
    for (word, tag) in at:
        if tag == 'NNP' and tag_prev == 'NNP':
            word_prev = word_prev + '_' + word
            continue

        if tag != 'NNP':
            if tag_prev == 'NNP': at_combinedNNP.append((word_prev, tag_prev))
            at_combinedNNP.append((word, tag))

        word_prev, tag_prev = word, tag
        
    return at_combinedNNP

#Preserve only nouns
def filterTag(targetTags, at):
    at_filteredTag = ((word, tag) for (word, tag) in at if tag in targetTags)
    return at_filteredTag

#Preserve only [a-zA-Z0-9_\-]
#Lower words as well
def filter8LowerWord(pattern, at):
    at_filteredWord = ((word.lower(), tag) for (word, tag) in at if re.search(pattern, word))
    return at_filteredWord

#Exclude specific words
def filterSWords(sWords, at):
    at_filteredSWords = ((word, tag) for (word, tag) in at if word not in sWords)
    return at_filteredSWords

#Combine POS and word
def appendTag(at):
    at_tagAppended = (word + '_' + tag for (word, tag) in at)
    return at_tagAppended

#Execution
sWords = ['https']
ats_filtered = [list(
    filterSWords(sWords,
    filter8LowerWord('^[a-zA-Z0-9_\-]+$',
    filterTag(['NN', 'NNP', 'NNS'],
    combineNNP(at
))))) for at in ats_tag]
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

ats_tfIdfFiltered = [list(filterTfIdf(at, 0.15, dct)) for at in ats_tfIdf]


#--Observation
def observe(idx):
    print(ats[idx])
    print('-' * 60)
    print(ats_tfIdfFiltered[idx])

observe(170)


#--Export
df_joined.to_csv('./data/df_joined.csv', index=False)
with open('./data/dict_category.json', 'w') as f:
    json.dump(dict_category, f)

#Test loading data back
pd.read_csv('./data/df_joined.csv', index_col=False).head()
with open('./data/dict_category.json', 'r') as f:
    print(json.load(f))