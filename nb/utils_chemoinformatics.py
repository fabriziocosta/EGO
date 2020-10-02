import numpy as np
import networkx as nx
import random
import logging
import matplotlib.pyplot as plt
from toolz import curry, pipe
from collections import defaultdict
from IPython.core.display import display

from eden.display import draw_graph, draw_graph_set, map_labels_to_colors
from eden.util import configure_logging
from eden_chem.io.pubchem import download
from eden_chem.io.rdkitutils import sdf_to_nx
from eden_chem.load_utils import pre_process, _random_sample
from eden_chem.io.pubchem import get_assay_description
from eden_chem.io.rdkitutils import nx_to_rdkit
from eden_chem.display.rdkitutils import nx_to_image

from ego.decomposition.paired_neighborhoods import decompose_paired_neighborhoods, decompose_neighborhood
from ego.vectorize import hash_graph
from ego.vectorize import set_feature_size, vectorize
from ego.encode import make_encoder

from utils_oracle_with_target import oracle_setup as oracle_setup_target
from utils_oracle_from_dataset import oracle_setup as oracle_setup_dataset
from eden_chem.io.rdkitutils import nx_to_inchi
from eden_chem.io.rdkitutils import nx_to_smi
from datetime import datetime



logger = logging.getLogger()
configure_logging(logger, verbosity=1)

download_active = curry(download)(active=True)
download_inactive = curry(download)(active=False)


def get_pos_graphs(assay_id): return pipe(assay_id, download_active, sdf_to_nx, list)


def get_neg_graphs(assay_id): return pipe(assay_id, download_inactive, sdf_to_nx, list)

colormap = 'tab20c'

#assay_ids = ['624466','492992','463230','651741','743219','588350','492952','624249','463213','2631','651610']

def rank_and_persist_molecules(graphs, scores, name='', plot=True):
    inchis = nx_to_inchi(graphs)
    smis = nx_to_smi(graphs)
    ids = sorted(range(len(scores)), key=lambda i:scores[i], reverse=True)
    
    sorted_graphs = [graphs[id] for id in ids] 
    sorted_scores = [scores[id] for id in ids]
    sorted_smis = [smis[id] for id in ids]
    sorted_inchis = [inchis[id] for id in ids]
    
    dt_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fname = '%s_constructed_%s.csv'%(name, dt_string)
    # save
    non_redundant_graphs = []
    previous_m = None
    i = 0
    with open(fname,'w') as file:
        txt = ('Rank\tScore\tCanonical Smiles\tInChI\tOrigin\n')
        file.write(txt)
        for s,m,inc,g in zip(sorted_scores, sorted_smis, sorted_inchis, sorted_graphs):
            if m != previous_m:
                txt = '%d\t%.3f\t%s\t%s\t%s'%(i+1, s, m, inc, g.graph.get('name',None))
                non_redundant_graphs.append(g)
                file.write(txt)
                if i<20:
                    print(txt)
                i = i+1
            previous_m = m
    # plot
    if plot:
        draw_graphs(non_redundant_graphs[:14])

def load_PUBCHEM_data(assay_id, max_size=20):
    configure_logging(logger, verbosity=2)
    logger.debug('_' * 80)
    logger.debug('Dataset %s info:' % assay_id)
    desc = get_assay_description(assay_id)
    logging.debug('\n%s' % desc)
    # extract pos and neg graphs
    all_pos_graphs, all_neg_graphs = get_pos_graphs(assay_id), get_neg_graphs(assay_id)
    # remove too large and too small graphs and outliers
    initial_max_size = 2000
    initial_max_size = max(initial_max_size, max_size)
    args = dict(initial_max_size=initial_max_size, fraction_to_remove=.1, n_neighbors_for_outliers=9, remove_similar=False, max_size=max_size)
    logging.debug('\nPositive graphs')
    pos_graphs = pre_process(all_pos_graphs, **args)
    logging.debug('\nNegative graphs')
    neg_graphs = pre_process(all_neg_graphs, **args)
    logger.debug('-' * 80)
    configure_logging(logger, verbosity=1)
    return pos_graphs, neg_graphs


def display_mol(graph, title, part_importance_estimator):
    node_score_dict, edge_score_dict = part_importance_estimator.predict(graph)
    weights = [node_score_dict[u] for u in graph.nodes()]
    mol = nx_to_rdkit(graph)
    fig = SimilarityMaps.GetSimilarityMapFromWeights(
        mol, weights, size=(250, 250), alpha=0.075, contourLines=1, sigma=.03)
    plt.title(title)
    plt.show()


def draw_graphs(graphs, titles=None, num=None, n_graphs_per_line=7):
    if titles is None:
        titles = [str(i) for i in range(len(graphs))]
    if num is not None:
        gs = graphs[:num]
        titles = titles[:num]
    else:
        gs = graphs
    for g, t in zip(gs, titles):
        g.graph['id'] = str(t)
    try:
        img = nx_to_image(gs, n_graphs_per_line=n_graphs_per_line, titles=titles)
        display(img)
    except Exception as e:
        colors = map_labels_to_colors(graphs)
        args = dict(layout='kk', colormap=colormap, vmin=0, vmax=1, vertex_size=80, edge_label=None,
                    vertex_color_dict=colors, vertex_color='-label-', vertex_label=None, ignore_for_layout='nesting')
        draw_graph_set(gs, n_graphs_per_line=6, size=7, **args)


def display_ktop_graphs(graphs, oracle_func, n_max=6):
    scores = [oracle_func(g) for g in graphs]
    ids = np.argsort(scores)[-n_max:]
    best_graphs = [graphs[id] for id in ids]
    best_scores = [scores[id] for id in ids]
    titles = ['%.3f' % best_score for best_score in best_scores]
    draw_graphs(best_graphs, titles=titles, n_graphs_per_line=6)


def remove_duplicates(graphs):
    df = decompose_neighborhood(radius=2)
    selected_graphs_dict = {hash_graph(
        g, decomposition_funcs=df): g for g in graphs}
    return list(selected_graphs_dict.values())


def target_quality(target_graph, graphs, max_score_threshold, min_score_threshold):
    # the quality of the target is measured as the fraction of graphs that are in a desired range of similarity
    oracle_func = oracle_setup_target(target_graph, random_noise=0.0)
    sel_graphs = [g for g in graphs if min_score_threshold < oracle_func(g) < max_score_threshold]
    quality_score = len(sel_graphs) / float(len(graphs))
    print('%.3f   ' % (quality_score))
    return quality_score


def build_chemical_experiment(assay_id, n_init_instances, n_domain_instances, max_score_threshold, n_targets=None, ratio_class_to_target=.9):
    pos_graphs, neg_graphs = load_PUBCHEM_data(assay_id, max_size=n_domain_instances)
    pos_graphs, neg_graphs = remove_duplicates(pos_graphs), remove_duplicates(neg_graphs) 
    domain_graphs = pos_graphs + neg_graphs
    random.shuffle(domain_graphs)

    if n_targets is None:
        oracle_func, target_graph = oracle_setup_dataset(
            pos_graphs, 
            neg_graphs, 
            ratio_class_to_target=ratio_class_to_target, 
            random_noise=0.0)
    else:
        print('Selecting a good cluster of molecular graphs in %d attempts. This might take a while...'%n_targets)
        target_graph = max(domain_graphs[:n_targets], key=lambda g: target_quality(g, domain_graphs, max_score_threshold, max_score_threshold / 1.5))
        oracle_func = oracle_setup_target(target_graph, random_noise=0.0)

    domain_graphs = [g for g in domain_graphs if oracle_func(g) < max_score_threshold]
    sorted_graphs = sorted(domain_graphs, key=lambda g: oracle_func(g), reverse=True)
    half_size = int(n_init_instances / 2)
    rest_graphs = sorted_graphs[half_size:]
    random.shuffle(rest_graphs)
    init_graphs = sorted_graphs[:half_size] + rest_graphs[:half_size]
    return init_graphs, domain_graphs, oracle_func, target_graph


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
    titles = ['%d) %.3f %s'%(len(history) - i,oracle_func(g),g.graph.get('type','orig').replace('Neighborhood', '')) for i,g in enumerate(history)]
    draw_graphs(history, titles)

def select_unique(codes, fragments):
    already_seen = set()
    unique_codes=[]
    unique_fragments=[]
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
        print('_'*80)
        draw_graphs([g],[''])
        codes, fragments = encoding_func(g)
        unique_codes, unique_fragments, code_counts = select_unique(codes, fragments)
        titles = ['%d   #%d'%(id,code_counts[id]) for id in unique_codes]
        print('%d unique components in %d fragments'%(len(unique_codes),len(codes)))
        if unique_fragments:
            draw_graphs(unique_fragments, titles, n_graphs_per_line=6)
        else:
            print('No fragments')