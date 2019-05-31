import networkx as nx
from infomap import infomap #https://mapequation.github.io/infomap/




#--Loading data back
G1 = nx.read_pajek("./data/G_description.net")


#--Community detection
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

tree_description = findCommunities(G1)


#--Export tree
