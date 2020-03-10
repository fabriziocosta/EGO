import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils_oracle_with_target import oracle_setup
from ego.vectorize import hash_graph
from ego.decomposition.paired_neighborhoods import decompose_paired_neighborhoods, decompose_neighborhood
import random
from toolz import curry 
from eden.display import draw_graph, draw_graph_set, map_labels_to_colors

from eden.util import configure_logging
import logging
logger = logging.getLogger()
configure_logging(logger, verbosity=1)

colormap='tab20c'


    
def draw_graphs(graphs, titles, n_graphs_per_line=6):
    size=np.log(len(graphs[0]))+1
    colors = map_labels_to_colors(graphs)
    gs = graphs[:]
    for g,t in zip(gs, titles): g.graph['id']=str(t)
    kwargs = dict(colormap=colormap, vertex_size=80, edge_label=None, vertex_color_dict=colors, vertex_color='_label_', vertex_label=None, layout='kk')
    draw_graph_set(gs, n_graphs_per_line=n_graphs_per_line, size=size, **kwargs)

    
def display_ktop_graphs(graphs, oracle_func, n_max=6):
    scores = [oracle_func(g) for g in graphs]
    ids = np.argsort(scores)[-n_max:]
    best_graphs = [graphs[id] for id in ids]
    best_scores = [scores[id] for id in ids]
    distinct_best_scores = []
    distinct_best_graphs = []
    prev_score = None
    counter = 0
    distinct_counters = []
    for best_graph, best_score in zip(best_graphs, best_scores):
        if prev_score != best_score:
            distinct_best_graphs.append(best_graph)
            distinct_best_scores.append(best_score)
            distinct_counters.append(counter)
            counter = 0
        else:
            counter += 1
        prev_score = best_score
    titles = ['%.6f x %d'%(distinct_best_scores[i], distinct_counters[i+1]+1) for i in range(len(distinct_best_scores)-1) ]
    titles += ['%.6f x %d'%(best_score,counter+1)]
    draw_graphs(distinct_best_graphs, titles=titles, n_graphs_per_line=6)
    
    
def make_instance(length=20, alphabet_size=3, frac=.3, start_char=97):
    n_frac=int(length*frac/(alphabet_size-1))

    def make_char(i,start_char=97):
        return chr(i+start_char)

    def make_chars(i, dim, start_char=97):
        return make_char(i,start_char)*dim

    line=''
    line += make_chars(0, length - n_frac*(alphabet_size-1), start_char)
    for i in range(1, alphabet_size):
        line += make_chars(i, n_frac, start_char)
    line=list(line)
    random.shuffle(line)
    return ''.join(line)

@curry
def random_path_graph(n):
    return nx.path_graph(n)

@curry
def random_tree_graph(n):
    return nx.random_tree(n)

@curry
def random_regular_graph(d, n):
    return nx.random_regular_graph(d,n)

@curry
def random_degree_seq(n, dmax):
    sequence = np.linspace(1, dmax,n).astype(int)
    return nx.expected_degree_graph(sequence)

@curry
def random_dense_graph(n, m):
    # a graph is chosen uniformly at random from the set of all graphs with n nodes and m edges
    g = nx.dense_gnm_random_graph(n, m)
    max_cc = max(nx.connected_components(g), key=lambda x: len(x))
    g = nx.subgraph(g, max_cc)
    return g

@curry
def make_graph(graph_generator, alphabet_size, frac):
    G = graph_generator
    labels = make_instance(length=len(G), alphabet_size=alphabet_size, frac=frac)
    dict_labels = {i:str(l) for i,l in enumerate(labels)}
    nx.set_node_attributes(G,dict_labels,'label')
    nx.set_edge_attributes(G,'1','label')
    return G

def make_sequence_data(target_graph, n_instances, diversity):
    # extract sequence of labels
    graphs = []
    for n in range(n_instances):
        seq = [target_graph.nodes[u]['label'] for u in target_graph.nodes()]
        for i in range(diversity):
            j = random.randint(1,len(target_graph))
            seq = seq[j:][::-1] + seq[:j]
        G=nx.path_graph(len(target_graph))
        dict_labels = {i:str(l) for i,l in enumerate(seq)}
        nx.set_node_attributes(G,dict_labels,'label')
        nx.set_edge_attributes(G,'1','label')
        graphs.append(G.copy())
    return graphs


def make_variants(target_graph):
    from ego.optimization.neighborhood_edge_swap import NeighborhoodEdgeSwap
    nes = NeighborhoodEdgeSwap(n_edges=2, n_neighbors=10)
    from ego.optimization.neighborhood_node_label_swap import NeighborhoodNodeLabelSwap
    nns = NeighborhoodNodeLabelSwap(n_nodes=1, n_neighbors=10)
    from ego.optimization.neighborhood_node_remove import NeighborhoodNodeRemove
    nnr = NeighborhoodNodeRemove(n_neighbors=10, n_nodes=1)

    gs = [target_graph]
    transformations = [nes, nns] # [nes, nns, nnr]
    for ne in transformations:
        gs = [ng for g in gs for ng in ne.neighbors(g)]
    return gs

def remove_duplicates(graphs):
    df = decompose_neighborhood(radius=2)
    selected_graphs_dict = {hash_graph(
        g, decomposition_funcs=df): g for g in graphs}
    return list(selected_graphs_dict.values())

def build_artificial_experiment(GRAPH_TYPE, n_init_instances, n_domain_instances, alphabet_size, diversity, max_score_threshold):
    if GRAPH_TYPE == 'path':
        graph_generator = random_path_graph(n=15)

    if GRAPH_TYPE == 'tree':
        graph_generator = random_tree_graph(n=18)
    
    if GRAPH_TYPE == 'degree':
        n=12
        dmax=4
        graph_generator = random_degree_seq(n, dmax)
        while nx.is_connected(graph_generator) is not True:
            graph_generator = random_degree_seq(n, dmax)
    
    if GRAPH_TYPE == 'regular':
        graph_generator = random_regular_graph(d=3, n=14)

    if GRAPH_TYPE == 'dense':
            graph_generator = random_dense_graph(n=15, m=38)
            
    target_graph = make_graph(graph_generator, alphabet_size=alphabet_size, frac=.5)
    domain_graphs = make_variants(target_graph)
    domain_graphs = domain_graphs[:n_domain_instances]
    oracle_func = oracle_setup(target_graph, random_noise=0.0)
    domain_graphs = [g for g in domain_graphs if oracle_func(g) < max_score_threshold]
    domain_graphs = remove_duplicates(domain_graphs)
    sorted_graphs = sorted(domain_graphs, key=lambda g:oracle_func(g), reverse=True) 
    half_size = int(n_init_instances/2)
    rest_graphs = sorted_graphs[half_size:]
    random.shuffle(rest_graphs)
    init_graphs = sorted_graphs[:half_size] + rest_graphs[:half_size]
    
    return init_graphs, domain_graphs, oracle_func, target_graph


def display_score_statistics(domain_graphs, oracle_func):
    n_plots=5
    plt.figure(figsize=(6,4))
    scores = np.array([oracle_func(g) for g in domain_graphs])
    plt.hist(scores, 30, density=True, alpha=.3)
    plt.title('Scores')
    plt.grid()
    plt.show()