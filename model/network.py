import pickle
import json
import numpy as np
import pandas as pd
import networkx as nx




#--Loading data back
#Main df
df = pd.read_csv('./data/df_joined.csv', index_col=False)
df.fillna('', inplace=True)
df.head()

#Category tag and description
with open('./data/dict_category.json', 'r') as f:
    cat = json.load(f)

#Tf-idf
with open('./data/ats_tfIdfFiltered.pkl', 'rb') as f:  
    ats_tfIdfFiltered = pickle.load(f)
    print(ats_tfIdfFiltered[0])


#--Dataset-node dict
mapping = dict(zip(G, string.ascii_lowercase))
mapping = df[['title']]


#--Graph 1: Description graph
#Transform tfidf data into dfs for merge
ats_dfs = [pd.DataFrame(at) for at in ats_tfIdfFiltered]

#Init edge matrix
edges = np.zeros((len(ats_dfs), len(ats_dfs)))

#Identify edges
for idx_i, at_i in enumerate(ats_dfs):
    if at_i.empty: continue #Skip empty df to speed up
    if idx_i == (len(ats_dfs) - 1): break #Special case for the last entry

    for idx_j, at_j in enumerate(ats_dfs[idx_i + 1:]):
        idx_j += idx_i + 1 #Use idx_i to skip half of the matrix, speeding up
        if at_j.empty: continue
        df_matchWord = pd.merge(ats_dfs[idx_i], ats_dfs[idx_j], how='inner', on=0)
        if not df_matchWord.empty:
            edges[idx_i][idx_j] = np.sum(df_matchWord[['1_x', '1_y']].values)

#Initiate graph from edge matrix
#TODO: Assign node weights according to the dataset popularity 
G1 = nx.from_numpy_array(edges, create_using=nx.Graph()) #Simple undirected graph

#Relabeling nodes
nx.relabel.relabel_nodes(G1, mapping, copy=False) #Relabel in place

#Observe
G1.number_of_nodes()
G1.number_of_edges()
G1.edges.data()


#--Graph 2: Tag graph


#--Export graphs
nx.write_pajek(G1, "G_description.net")

#Test loading graph back
G1 = nx.read_pajek("G_description.net")