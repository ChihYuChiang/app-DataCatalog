import re
import networkx as nx
from infomap import infomap #https://mapequation.github.io/infomap/




#--Loading data back
#Graph and mapping
G1 = nx.read_pajek("./data/G_description.net")
with open('./data/mapping.pkl', 'rb') as f:  
    mapping = pickle.load(f)

#Tree
with open("./data/G_description.ftree", "r") as f:
    tree1 = f.read()


#--Community detection
#Currently, using the online version to produce the `ftree` file
#https://github.com/mapequation/infomap/blob/master/examples/python/infomap-examples.ipynb
def findCommunities(G):
    """
    Partition network with the Infomap algorithm.
    Annotates nodes with 'community' id and return number of communities found.
    """
    infomapWrapper = infomap.Infomap("--two-level --silent")

    #Building Infomap network from a NetworkX graph
    for e in G.edges():
        infomapWrapper.addLink(*e)

    #Find communities with Infomap
    infomapWrapper.run()
    tree = infomapWrapper.tree

    #Community description
    print("Found %d modules with codelength: %f" % (tree.numTopModules(), tree.codelength()))

    return tree

tree1 = findCommunities(G1)


#--Relabeling ftree
#ftree is plain string
for k in mapping.keys():
    tree1 = re.sub('"{}"'.format(str(k + 1)), '"{}"'.format(mapping[k]), tree1)

#Visualization
#Note: "Trump XXX" could cause quotation trailing error when upload to the navigator server and could require manual removal.
#https://www.mapequation.org/navigator/

#Write to file
with open("./data/G_description_labeled.ftree", "w") as f:
    f.write(tree1)
