#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
from ego.predictor import Classifier
from sklearn.metrics import roc_auc_score
from toolz import curry
from ego.utils.timeout_utils import assign_timeout
from ego.optimization.hyper_optimize import serialize_tree, hyper_optimize_decomposition_function
from ego.vectorize import vectorize
from ego.learn import PartImportanceEstimator

# order 0
from ego.decomposition.identity import decompose_identity
from ego.decomposition.nodes_edges import decompose_nodes_and_edges, decompose_nodes, decompose_edges
from ego.decomposition.path import decompose_path
from ego.decomposition.paired_neighborhoods import decompose_paired_neighborhoods, decompose_neighborhood
from ego.decomposition.cycle_basis import decompose_cycles_and_non_cycles, decompose_non_cycles, decompose_cycles
from ego.decomposition.clique import decompose_clique_and_non_clique, decompose_clique, decompose_non_clique
from ego.decomposition.graphlet import decompose_graphlet

# node-edge filter based
from ego.decomposition.degree import decompose_degree_and_non_degree, decompose_degree, decompose_non_degree
from ego.decomposition.centrality import decompose_central_and_non_central, decompose_central, decompose_non_central
from ego.decomposition.positive_and_negative import decompose_positive, decompose_negative, decompose_positive_and_negative

# order 1
from ego.decomposition.size import decompose_node_size, decompose_edge_size
from ego.decomposition.context import decompose_context
from ego.decomposition.dilatate import decompose_dilatate
#from ego.decomposition.union import decompose_union
from ego.decomposition.join import decompose_node_join, decompose_edge_join
from ego.decomposition.pair import decompose_pair

from ego.decomposition.iterated_clique import decompose_iterated_clique

# order 2
from ego.decomposition.set import decompose_difference, decompose_symmetric_difference, decompose_union, decompose_intersection
from ego.decomposition.relation import decompose_relation
from ego.decomposition.pair_binary import decompose_pair_binary

# order 3
from ego.decomposition.relation_binary import decompose_relation_binary

# order n
from ego.decomposition.concatenate import decompose_concatenate


def explain(history):
    part_dict = dict()
    for thash in history:
        tree, auc, comp = history[thash]
        str_tree = serialize_tree(tree)
        str_tree = str_tree.replace('(', '')
        str_tree = str_tree.replace(')', '')
        str_tree = str_tree.replace('|', ' ')
        parts = str_tree.split()
        for part in parts:
            if part not in part_dict:
                part_dict[part] = [auc]
            else:
                part_dict[part].append(auc)
    for part in sorted(part_dict, key=lambda part: np.mean(part_dict[part]) - np.std(part_dict[part]), reverse=True):
        m = np.mean(part_dict[part])
        s = np.std(part_dict[part])
        print('%10s   %.2f +- %.2f' % (part, m, s))


def _evaluate(decompose_func, train_graphs=None, train_target=None, test_graphs=None, test_target=None, n_iter=3):
    scores = []
    for i in range(n_iter):
        disc = Classifier(decompose_func=decompose_func,
                          preprocessor=None, nbits=14, seed=i)
        disc.fit(train_graphs, train_target)
        preds = disc.decision_function(test_graphs)
        score = roc_auc_score(test_target, preds)
        scores.append(score)
    return np.mean(scores) - np.std(scores) * 0.1


@curry
def evaluate(decompose_func, train_graphs=None, train_target=None, test_graphs=None, test_target=None, n_iter=3, timeout=10):
    _evaluate_ = assign_timeout(_evaluate, timeout)
    r = _evaluate_(decompose_func, train_graphs, train_target,
                   test_graphs, test_target, n_iter)
    if r is None:
        return 0.5
    else:
        return r


def _evaluate_complexity(decompose_func, graphs=None):
    D = vectorize(graphs, decomposition_funcs=decompose_func)
    K = D.dot(D.T).todense()
    eig_vals, eig_vecs = np.linalg.eig(K)
    v = eig_vals * eig_vals
    v = -np.sort(-v)
    v = np.log(v + 1)
    d = np.mean(v)
    return d


@curry
def evaluate_complexity(decompose_func, graphs=None, timeout=10):
    _evaluate_complexity_ = assign_timeout(_evaluate_complexity, timeout)
    r = _evaluate_complexity_(decompose_func, graphs)
    if r is None:
        return 10
    else:
        return r


def make_evaluate_identity(graphs):
    ids = np.random.choice(len(graphs), min(20, len(graphs)), replace=False)
    sel_graphs = [graphs[id] for id in ids]

    def evaluate_identity(decompose_func):
        D = vectorize(sel_graphs, decomposition_funcs=decompose_func)
        signature = (D.count_nonzero(), D.sum())
        return signature

    return evaluate_identity


def make_data_split(pos_graphs, neg_graphs, test_frac=.3):
    test_set_size = int(len(pos_graphs) * test_frac)
    test_pos_graphs = pos_graphs[:test_set_size]
    train_pos_graphs = pos_graphs[test_set_size:]
    test_set_size = int(len(neg_graphs) * test_frac)
    test_neg_graphs = neg_graphs[:test_set_size]
    train_neg_graphs = neg_graphs[test_set_size:]
    train_graphs = train_pos_graphs + train_neg_graphs
    train_target = [1] * len(train_pos_graphs) + [-1] * len(train_neg_graphs)
    test_graphs = test_pos_graphs + test_neg_graphs
    test_target = [1] * len(test_pos_graphs) + [-1] * len(test_neg_graphs)
    return train_graphs, train_target, test_graphs, test_target


def make_decomposition_dict():
    fs = []
    for i in range(10):
        decomposition_functions = dict()
        decomposition_functions['+'] = decompose_concatenate
        fs.append(decomposition_functions)
    fs[0].pop('+')
    fs[1].pop('+')

    fs[0]['edg'] = decompose_edges
    fs[0]['cyc'] = decompose_cycles
    fs[0]['cyc&n'] = decompose_cycles_and_non_cycles
    fs[0]['ncyc'] = decompose_non_cycles
    fs[0]['nb'] = decompose_neighborhood(radius=1)
    fs[0]['nb2'] = decompose_neighborhood(radius=2)
    fs[0]['nb3'] = decompose_neighborhood(radius=3)
    #fs[0]['grfl3'] = decompose_graphlet(size=3)
    #fs[0]['grfl4'] = decompose_graphlet(size=4)
    #fs[0]['grfl5'] = decompose_graphlet(size=5)
    fs[0]['centr5'] = decompose_central(k_top=5)
    fs[0]['centr7'] = decompose_central(k_top=7)
    fs[0]['centr9'] = decompose_central(k_top=9)
    fs[0]['centr11'] = decompose_central(k_top=11)
    fs[0]['centr13'] = decompose_central(k_top=13)
    fs[0]['ncentr5'] = decompose_non_central(k_top=5)
    fs[0]['ncentr7'] = decompose_non_central(k_top=7)
    fs[0]['ncentr9'] = decompose_non_central(k_top=9)
    fs[0]['ncentr11'] = decompose_non_central(k_top=11)
    fs[0]['ncentr13'] = decompose_non_central(k_top=13)

    fs[1]["nb'"] = decompose_neighborhood(radius=1)
    fs[1]["nb2'"] = decompose_neighborhood(radius=2)
    fs[1]["nb''"] = decompose_neighborhood(radius=1)
    fs[1]["nb2''"] = decompose_neighborhood(radius=2)
    fs[1]['dlt'] = decompose_dilatate(radius=1)
    fs[1]['dlt2'] = decompose_dilatate(radius=2)
    fs[1]['dlt3'] = decompose_dilatate(radius=3)
    fs[1]['cntx'] = decompose_context(radius=1)
    fs[1]['cntx2'] = decompose_context(radius=2)
    fs[1]['cntx3'] = decompose_context(radius=3)
    fs[1]['njn'] = decompose_node_join
    fs[1]['ejn'] = decompose_edge_join
    fs[1]['pr1'] = decompose_pair(distance=1)
    fs[1]['pr2'] = decompose_pair(distance=2)
    fs[1]['pr3'] = decompose_pair(distance=3)
    fs[1]['sz<5'] = decompose_node_size(max_size=5)
    fs[1]['sz<7'] = decompose_node_size(max_size=7)
    fs[1]['sz<9'] = decompose_node_size(max_size=9)
    fs[1]['sz<11'] = decompose_node_size(max_size=11)
    fs[1]['sz<13'] = decompose_node_size(max_size=13)
    fs[1]['sz>5'] = decompose_node_size(min_size=5)
    fs[1]['sz>7'] = decompose_node_size(min_size=7)
    fs[1]['sz>9'] = decompose_node_size(min_size=9)
    fs[1]['sz>11'] = decompose_node_size(min_size=11)
    fs[1]['sz>13'] = decompose_node_size(min_size=13)

    fs[2]['unn'] = decompose_union
    fs[2]['int'] = decompose_intersection
    fs[2]['dff'] = decompose_difference
    fs[2]['sdff'] = decompose_symmetric_difference
    fs[2]['rel'] = decompose_relation
    fs[2]['pair1'] = decompose_pair_binary(distance=1)
    fs[2]['pair2'] = decompose_pair_binary(distance=2)
    fs[2]['pair3'] = decompose_pair_binary(distance=3)

    fs[3]['relbi'] = decompose_relation_binary(keep_second_component=False)
    fs[3]['relbi2nd'] = decompose_relation_binary(keep_second_component=True)

    return fs


def hyper_optimize_setup(pos_graphs, neg_graphs, timeout=60, test_frac=.50, n_iter=3, add_data_driven=False):
    train_graphs, train_target, test_graphs, test_target = make_data_split(
        pos_graphs, neg_graphs, test_frac=test_frac)
    evaluate_func = evaluate(train_graphs=train_graphs, train_target=train_target, test_graphs=test_graphs,
                             test_target=test_target, n_iter=n_iter, timeout=timeout)
    evaluate_complexity_func = evaluate_complexity(
        graphs=train_graphs, timeout=timeout)
    evaluate_identity_func = make_evaluate_identity(graphs=train_graphs)
    decomposition_dict = make_decomposition_dict()

    if add_data_driven:
        decomposition_dict[0]['adpts11nb1'] = decompose_positive(ktop=11, part_importance_estimator=PartImportanceEstimator(
            decompose_neighborhood(radius=1)).fit(train_graphs, train_target))
        decomposition_dict[0]['adpts7nb1'] = decompose_positive(ktop=7, part_importance_estimator=PartImportanceEstimator(
            decompose_neighborhood(radius=1)).fit(train_graphs, train_target))

        decomposition_dict[0]['adpts11nb2'] = decompose_positive(ktop=11, part_importance_estimator=PartImportanceEstimator(
            decompose_neighborhood(radius=2)).fit(train_graphs, train_target))
        decomposition_dict[0]['adpts7nb2'] = decompose_positive(ktop=7, part_importance_estimator=PartImportanceEstimator(
            decompose_neighborhood(radius=2)).fit(train_graphs, train_target))
    return (train_graphs, train_target, test_graphs, test_target), evaluate_func, evaluate_complexity_func, evaluate_identity_func, decomposition_dict


def hyperopt(pos_graphs, neg_graphs, data_size,
             memory, history, auc_threshold, cmpx_threshold,
             n_max, n_max_sel, n_max_sample, order,
             timeout, max_n_hours, max_runtime, test_frac, n_iter, display=True, add_data_driven=False):
    hyper_params = hyper_optimize_setup(
        pos_graphs, neg_graphs, timeout, test_frac=test_frac, n_iter=n_iter, add_data_driven=add_data_driven)
    data_partition, evaluate_func, evaluate_complexity_func, evaluate_identity_func, decomposition_dict = hyper_params
    train_graphs, train_target, test_graphs, test_target = data_partition

    opt_decompose_func, opt_decompose_func_str = hyper_optimize_decomposition_function(
        order, decomposition_dict,
        evaluate_func, evaluate_complexity_func, evaluate_identity_func,
        n_max, n_max_sel, n_max_sample, auc_threshold, cmpx_threshold,
        memory, history,
        max_runtime=max_runtime, display=display,
        return_decomposition_function_string=True)
    if opt_decompose_func is None:
        return decompose_neighborhood(radius=2), 'nb2'
    return opt_decompose_func, opt_decompose_func_str
