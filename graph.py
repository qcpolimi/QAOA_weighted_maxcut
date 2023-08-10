import networkx as nx
import numpy as np


def set_weights(G, w, weight_dist, seed):
    if w is None:
        return G
    np.random.seed(seed)

    n_edges = G.number_of_edges()

    if len(w) == 2:
        if w[0] == w[1]:
            return G

    if weight_dist == 'fixed':
        assert len(w) == n_edges, "w list needs to specify weights for all nodes"
        weights = w
    elif weight_dist == 'uniform':
        # w[0] = lower bound, w[1] = higher bound
        weights = np.random.uniform(w[0], w[1], n_edges)
    elif weight_dist == 'gaussian':
        # w[0] = mean, w[1] = stand. deviation
        weights = np.random.normal(w[0], w[1], n_edges)
    # elif weight_dist == 'exponential':
    #     # w is the scale parameter `w = 1/lambda`
    #     weights = np.random.exponential(w, n_edges)

    for i, (u, v) in enumerate(G.edges()):
        G[u][v]['weight'] = weights[i]
    return G


def directed(n, k, alpha=3, seed=1, self_loops=False):
    G = nx.random_k_out_graph(n, k, alpha=alpha, seed=seed,self_loops=self_loops)
    assert nx.is_connected(G), "Graph not connected, change seed"
    return G
    

def fully_connected(n, w=None, weight_dist='uniform', seed=1):
    G = nx.complete_graph(n)
    G = set_weights(G, w, weight_dist, seed)
    return G


def closed_cycle(n, w=None, weight_dist='uniform', seed=1):
    G = nx.cycle_graph(n)
    G = set_weights(G, w, weight_dist, seed)
    return G


def cycle(n, w=None, weight_dist='uniform', seed=1):
    G = nx.cycle_graph(n)
    G = set_weights(G, w, weight_dist, seed)
    G.remove_edge(0, n-1)
    return G


def star(n, w=None, weight_dist='uniform', seed=1):
    G = nx.star_graph(n-1)
    G = set_weights(G, w, weight_dist, seed)
    return G


def erdos_renyi(n, p, w=None, weight_dist='uniform', seed=1):
    G = nx.generators.random_graphs.erdos_renyi_graph(n, p, seed=seed)
    assert nx.is_connected(G), "Graph not connected, change seed"
    G = set_weights(G, w, weight_dist, seed)
    return G


def barabasi_albert(n, m, w=None, weight_dist='uniform', seed=1):
    G = nx.generators.random_graphs.barabasi_albert_graph(n, m, seed=seed)
    assert nx.is_connected(G), "Graph not connected, change seed"
    G = set_weights(G, w, weight_dist, seed)
    return G
    
    
def draw_graph(G, cut=False):
    pos = nx.circular_layout(G)

    color_dict = {k:v for k,v in zip(G.nodes(), cut)} if cut else {k:'0'*len(G.nodes()) for k in G.nodes()}
    nx.draw_networkx_nodes(G, pos, node_color=get_color_map(color_dict))
    nx.draw_networkx_edges(G, pos, width=1)
    nx.draw_networkx_labels(G, pos, font_size=12)

    if nx.is_weighted(G) == True:
        edge_labels = nx.get_edge_attributes(G, "weight") 
        for k, v in edge_labels.items():
            edge_labels[k] = np.round(v, 2)
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=12)
   

def get_color_map(color_dict):
    color_map = []
    for k,v in sorted(color_dict.items()):
        if v == '0':
            color_map.append("#DDDDDD")
        elif v == '1':
            color_map.append("#888888")
        else:
            color_map.append("green")
    return color_map


# def get_A_matrix(G):
#     n = G.number_of_nodes()
#     A = np.zeros([n, n])
#     for u, v, d in G.edges(data=True):
#         if nx.is_directed 
#         if nx.is_weighted(G):
#             A[u, v] = d["weight"]
#             A[v, u] = d["weight"]
#         else:
#             A[u, v] = 1
#             A[v, u] = 1
#     return A


def get_Q_matrix(G):
    A = nx.adjacency_matrix(G)
    n = A.shape[0]
    Q = np.zeros([n, n])
    for i in range(n):
        x = 0
        for j in range(n):
            if i != j:
                Q[i, j] = A[i, j]   
                x += -A[i, j]
        Q[i, i] = x
    return Q