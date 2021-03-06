import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils_oracle_with_target import oracle_setup
from ego.vectorize import hash_graph
from ego.decomposition.paired_neighborhoods import decompose_paired_neighborhoods, decompose_neighborhood
import random
from collections import defaultdict
from toolz import curry
from eden.display import draw_graph, draw_graph_set, map_labels_to_colors

from ego.vectorize import set_feature_size, vectorize
from ego.encode import make_encoder

from ego.optimization.neighborhood_edge_swap import NeighborhoodEdgeSwap
from ego.optimization.neighborhood_edge_label_swap import NeighborhoodEdgeLabelSwap
from ego.optimization.neighborhood_edge_label_mutation import NeighborhoodEdgeLabelMutation
from ego.optimization.neighborhood_edge_move import NeighborhoodEdgeMove
from ego.optimization.neighborhood_edge_remove import NeighborhoodEdgeRemove
from ego.optimization.neighborhood_edge_add import NeighborhoodEdgeAdd
from ego.optimization.neighborhood_edge_expand import NeighborhoodEdgeExpand
from ego.optimization.neighborhood_edge_contract import NeighborhoodEdgeContract

from ego.optimization.neighborhood_node_label_swap import NeighborhoodNodeLabelSwap
from ego.optimization.neighborhood_node_label_mutation import NeighborhoodNodeLabelMutation
from ego.optimization.neighborhood_node_remove import NeighborhoodNodeRemove
from ego.optimization.neighborhood_node_add import NeighborhoodNodeAdd
from ego.optimization.neighborhood_node_smooth import NeighborhoodNodeSmooth

from eden.util import configure_logging
import logging
logger = logging.getLogger()
configure_logging(logger, verbosity=1)

colormap = 'tab20c'

def plot_histogram(data, title, max_num=30):
    if len(data) == 0:
        return
    if max(data) - min(data)>max_num:
        binwidth=(max(data) - min(data))/max_num
        rotation=45
    else:
        binwidth=1
        rotation=0
    bins=np.arange(min(data), max(data) + binwidth, binwidth)
    
    plt.hist(data, bins=bins, log=True, alpha=.3, density=True, rwidth=0.85, edgeColor='k')
    plt.title(title)
    plt.xticks(bins, rotation=rotation)
    plt.grid(axis='y')

def get_cycle_lenghts(graphs):
    all_measures = []
    for g in graphs:
        measures = [len(cyc) for cyc in nx.cycle_basis(g)]
        all_measures.extend(measures)
    return all_measures

def get_degree(graphs):
    all_measures = []
    for g in graphs:
        measures = [g.degree(u) for u in g.nodes()]
        all_measures.extend(measures)
    return all_measures

def get_betweenness_centrality(graphs):
    all_measures = []
    for g in graphs:
        measures = [val*100 for val in nx.betweenness_centrality(g, normalized=True).values()]
        all_measures.extend(measures)
    return all_measures

def get_num_nodes_and_edges(graphs):
    num_nodes = np.array([g.number_of_nodes() for g in graphs])
    num_edges = np.array([g.number_of_edges() for g in graphs])
    return np.mean(num_nodes), np.std(num_nodes), np.mean(num_edges), np.std(num_edges)
    
def plot_data_stats(graphs, name=None, size=5):
    plt.figure(figsize=(size*5, size))
    plt.subplot(131)
    plot_histogram(get_cycle_lenghts(graphs), 'Cycle length')
    plt.subplot(132)
    plot_histogram(get_degree(graphs), 'Degree')
    plt.subplot(133)
    plot_histogram(get_betweenness_centrality(graphs), 'Centrality')
    n_mu,n_std,e_mu,e_std = get_num_nodes_and_edges(graphs)
    plt.suptitle('Statistics for %s (%d graphs with avg #nodes:%.1f +- %.1f  #edges:%.1f +- %.1f)'%(name, len(graphs), n_mu, n_std, e_mu, e_std))
    plt.show()
    
def draw_graphs(graphs, titles=None, n_graphs_per_line=6):
    size = np.log(len(graphs[0])) + 1
    colors = map_labels_to_colors(graphs)
    gs = graphs[:]
    if titles is None:
        for i, g in enumerate(gs):
            g.graph['id'] = str(i)
    else:
        for g, t in zip(gs, titles):
            g.graph['id'] = str(t)
    kwargs = dict(colormap=colormap, vertex_size=80, edge_label=None, vertex_color_dict=colors, vmax=1, vmin=0,
                  vertex_color='_label_', vertex_label=None, layout='kk', ignore_for_layout='nesting')
    if len(colors) == 1:
        kwargs['vertex_fixed_color']='w'
        kwargs['vertex_color']=None
        kwargs['vertex_alpha']=0.8
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
    titles = ['%.6f x %d' % (distinct_best_scores[i], distinct_counters[i + 1] + 1) for i in range(len(distinct_best_scores) - 1)]
    titles += ['%.6f x %d' % (best_score, counter + 1)]
    draw_graphs(distinct_best_graphs, titles=titles, n_graphs_per_line=6)


def make_instance(length=20, alphabet_size=3, frac=.3, start_char=97):
    if alphabet_size == 1:
        ch = chr(start_char)
        line = [ch] * length
        return ''.join(line)

    n_frac = int(length * frac / (alphabet_size - 1))

    def make_char(i, start_char=97):
        return chr(i + start_char)

    def make_chars(i, dim, start_char=97):
        return make_char(i, start_char) * dim

    line = ''
    line += make_chars(0, length - n_frac * (alphabet_size - 1), start_char)
    for i in range(1, alphabet_size):
        line += make_chars(i, n_frac, start_char)
    remainder = length - len(line)
    if remainder > 0:
        line += make_char(0, start_char) * remainder
    line = list(line)
    random.shuffle(line)
    return ''.join(line)


@curry
def random_path_graph(n):
    return nx.path_graph(n)


@curry
def random_tree_graph(n):
    return nx.random_tree(n)


@curry
def random_cycle_graph(n):
    G = nx.random_tree(n)
    terminals = [u for u in G.nodes()if G.degree(u) == 1]
    random.shuffle(terminals)
    for i in range(0, len(terminals), 2):
        e_start = terminals[i]
        if i + 1 < len(terminals):
            e_end = terminals[i + 1]
            G.add_edge(e_start, e_end)
    return G


@curry
def random_regular_graph(d, n):
    return nx.random_regular_graph(d, n)


@curry
def random_degree_seq(n, dmax):
    sequence = np.linspace(1, dmax, n).astype(int)
    return nx.expected_degree_graph(sequence)


@curry
def random_dense_graph(n, m):
    # a graph is chosen uniformly at random from the set of all graphs with n nodes and m edges
    g = nx.dense_gnm_random_graph(n, m)
    max_cc = max(nx.connected_components(g), key=lambda x: len(x))
    g = nx.subgraph(g, max_cc)
    g = nx.convert_node_labels_to_integers(g)
    return g


@curry
def make_graph(graph_generator, alphabet_size, frac):
    G = graph_generator
    labels = make_instance(length=len(G), alphabet_size=alphabet_size, frac=frac)
    dict_labels = {i: str(l) for i, l in enumerate(labels)}
    assert len(dict_labels) == len(G)
    nx.set_node_attributes(G, dict_labels, 'label')
    nx.set_edge_attributes(G, '1', 'label')
    return G


def make_sequence_data(target_graph, n_instances, diversity):
    # extract sequence of labels
    graphs = []
    for n in range(n_instances):
        seq = [target_graph.nodes[u]['label'] for u in target_graph.nodes()]
        for i in range(diversity):
            j = random.randint(1, len(target_graph))
            seq = seq[j:][::-1] + seq[:j]
        G = nx.path_graph(len(target_graph))
        dict_labels = {i: str(l) for i, l in enumerate(seq)}
        nx.set_node_attributes(G, dict_labels, 'label')
        nx.set_edge_attributes(G, '1', 'label')
        graphs.append(G.copy())
    return graphs


def make_variants(target_graph=None, n_iterations=2, n_neighbors=2, neighborhood_estimators=None):
    graphs = [target_graph]

    transformations = []
    for neighborhood_estimator in neighborhood_estimators:
        if neighborhood_estimator == 'edge_swap':
            transformations.append(NeighborhoodEdgeSwap(n_neighbors=n_neighbors).fit(graphs, None))
        if neighborhood_estimator == 'edge_move':
            transformations.append(NeighborhoodEdgeMove(n_neighbors=n_neighbors).fit(graphs, None))
        if neighborhood_estimator == 'edge_remove':
            transformations.append(NeighborhoodEdgeRemove(n_neighbors=n_neighbors).fit(graphs, None))
        if neighborhood_estimator == 'edge_add':
            transformations.append(NeighborhoodEdgeAdd(n_neighbors=n_neighbors).fit(graphs, None))
        if neighborhood_estimator == 'edge_expand':
            transformations.append(NeighborhoodEdgeExpand(n_neighbors=n_neighbors).fit(graphs, None))
        if neighborhood_estimator == 'edge_contract':
            transformations.append(NeighborhoodEdgeContract(n_neighbors=n_neighbors).fit(graphs, None))

        if neighborhood_estimator == 'node_label_swap':
            transformations.append(NeighborhoodNodeLabelSwap(n_neighbors=n_neighbors).fit(graphs, None))
        if neighborhood_estimator == 'node_label_mutation':
            transformations.append(NeighborhoodNodeLabelMutation(n_neighbors=n_neighbors).fit(graphs, None))
        if neighborhood_estimator == 'node_add':
            transformations.append(NeighborhoodNodeAdd(n_neighbors=n_neighbors).fit(graphs, None))
        if neighborhood_estimator == 'node_remove':
            transformations.append(NeighborhoodNodeRemove(n_neighbors=n_neighbors).fit(graphs, None))
        if neighborhood_estimator == 'node_smooth':
            transformations.append(NeighborhoodNodeSmooth(n_neighbors=n_neighbors).fit(graphs, None))

    all_graphs = []
    for iter in range(n_iterations):
        for neighborhood_estimator in transformations:
            graphs = [neighbor_graph for g in graphs for neighbor_graph in neighborhood_estimator.neighbors(g)]
            all_graphs.extend(graphs[:])
            print(len(all_graphs))
    return all_graphs


def remove_duplicates(graphs):
    df = decompose_neighborhood(radius=2)
    selected_graphs_dict = {hash_graph(
        g, decomposition_funcs=df): g for g in graphs}
    return list(selected_graphs_dict.values())


def build_artificial_experiment(GRAPH_TYPE='degree', instance_size=20, n_init_instances=10, n_domain_instances=100, alphabet_size=4, max_score_threshold=.8, max_num_results=None, oracle_func=None, neighborhood_estimators=['edge_swap', 'node_label_swap'], n_iterations=4, n_neighbors_per_estimator_per_iteration=2):
    if GRAPH_TYPE == 'path':
        graph_generator = random_path_graph(n=instance_size)

    if GRAPH_TYPE == 'tree':
        graph_generator = random_tree_graph(n=instance_size)

    if GRAPH_TYPE == 'cycle':
        graph_generator = random_cycle_graph(n=instance_size)

    if GRAPH_TYPE == 'degree':
        n = instance_size
        dmax = 4
        graph_generator = random_degree_seq(n, dmax)
        while nx.is_connected(graph_generator) is not True:
            graph_generator = random_degree_seq(n, dmax)

    if GRAPH_TYPE == 'regular':
        graph_generator = random_regular_graph(d=3, n=instance_size)

    if GRAPH_TYPE == 'dense':
        graph_generator = random_dense_graph(n=instance_size, m=instance_size + instance_size // 2)

    target_graph = make_graph(graph_generator, alphabet_size=alphabet_size, frac=.5)
    if oracle_func is None:
        oracle_func = oracle_setup(target_graph, random_noise=0.0)
    domain_graphs = make_variants(target_graph, n_iterations, n_neighbors_per_estimator_per_iteration, neighborhood_estimators)
    random.shuffle(domain_graphs)
    domain_graphs = domain_graphs[:n_domain_instances]
    print('Generated: %d graphs'%len(domain_graphs))

    domain_graphs = remove_duplicates(domain_graphs)
    print('After duplicate removal: %d graphs'%len(domain_graphs))

    if max_num_results is not None:
        sorted_graphs = sorted(domain_graphs, key=lambda g: oracle_func(g), reverse=False)
        init_graphs = sorted_graphs[:n_init_instances]
        print('After sorting: %d graphs'%len(init_graphs))
    else:
        domain_graphs = [g for g in domain_graphs if oracle_func(g) < max_score_threshold]
        print('After max score threshold: %d graphs'%len(domain_graphs))
        random.shuffle(domain_graphs)
        # if simple  ouptut is needed
        init_graphs = domain_graphs[:n_init_instances]
    return init_graphs, domain_graphs, oracle_func, target_graph

    #if more accurate ouptut is needed
    sorted_graphs = sorted(domain_graphs, key=lambda g: oracle_func(g), reverse=True)
    half_size = int(n_init_instances / 2)
    rest_graphs = sorted_graphs[half_size:]
    random.shuffle(rest_graphs)
    init_graphs = sorted_graphs[:half_size] + rest_graphs[:half_size]
    return init_graphs, sorted_graphs, oracle_func, target_graph


def display_score_statistics(domain_graphs, oracle_func):
    n_plots = 5
    plt.figure(figsize=(6, 4))
    scores = np.array([oracle_func(g) for g in domain_graphs])
    plt.hist(scores, 30, density=True, alpha=.3)
    plt.title('Scores')
    plt.grid()
    plt.show()


def draw_history(graphs, oracle_func):
    best_graph = max(graphs, key=lambda g: oracle_func(g))
    history = [best_graph]
    while True:
        g = history[-1]
        parent = g.graph.get('parent', None)
        if parent is None:
            break
        history.append(parent)
    titles = ['%d) %.3f %s' % (len(history) - i, oracle_func(g), g.graph.get('type', 'orig').replace('Neighborhood', ''))
              for i, g in enumerate(history)]
    draw_graphs(history, titles)


def select_unique(codes, fragments):
    already_seen = set()
    unique_codes = []
    unique_fragments = []
    code_counts = defaultdict(int)
    for code, fragment in zip(codes, fragments):
        if code not in already_seen:
            unique_codes.append(code)
            unique_fragments.append(fragment)
            already_seen.add(code)
        code_counts[code] += 1
    return unique_codes, unique_fragments, code_counts


def draw_decomposition_graphs(graphs, decompose_funcs, preprocessors=None, draw_graphs=None):
    feature_size, bitmask = set_feature_size(nbits=14)
    encoding_func = make_encoder(decompose_funcs, preprocessors=preprocessors, bitmask=bitmask, seed=1)
    for g in graphs:
        print('_' * 80)
        draw_graphs([g], [''])
        codes, fragments = encoding_func(g)
        unique_codes, unique_fragments, code_counts = select_unique(codes, fragments)
        titles = ['%d   #%d' % (id, code_counts[id]) for id in unique_codes]
        print('%d unique components in %d fragments' % (len(unique_codes), len(codes)))
        if unique_fragments:
            draw_graphs(unique_fragments, titles, n_graphs_per_line=6)
        else:
            print('No fragments')
