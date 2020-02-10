#!/usr/bin/env python
"""Provides scikit interface."""
import signal
import time
import numpy as np
import networkx as nx
from ego.pareto_utils import pareto_select
from ego.decompose import concatenate, compose
from ego.size import decompose_node_size
import seaborn as sns
import itertools
import matplotlib.pyplot as plt


import logging
logger = logging.getLogger()


def serialize_tree(tree, node_id=0):
    s = tree.nodes[node_id]['label']
    n_neighbors = len(list(tree.neighbors(node_id)))
    if n_neighbors > 0:
        s += '('
        for i, u in enumerate(tree.neighbors(node_id)):
            s += serialize_tree(tree, u)
            if i < n_neighbors - 1:
                s += ','
        s += ')'
    return s


def serialize_tree_structure(tree, node_id=0):
    s = 'O('
    n_neighbors = len(list(tree.neighbors(node_id)))
    for i, u in enumerate(tree.neighbors(node_id)):
        s += serialize_tree(tree, u)
        if i < n_neighbors - 1:
            s += '|'
    s += ')'
    return s


def hash_tree_structure(tree, node_id=0, nbits=14):
    bitmask = pow(2, nbits) - 1
    code = hash(serialize_tree_structure(tree, node_id)) & bitmask
    return code


def hash_tree(tree, node_id=0, nbits=14):
    bitmask = pow(2, nbits) - 1
    code = hash(serialize_tree(tree, node_id)) & bitmask
    return code


def assign_timeout(func, timeout):

    def handler(signum, frame):
        raise Exception("end of timeout")

    def timed_func(*args, **kargs):
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        try:
            r = func(*args, **kargs)
            return r
        except Exception:
            return None
    return timed_func


def _get_decompose_func(tree, node_id=0):
    n_neighbors = len(list(tree.neighbors(node_id)))
    if n_neighbors > 0:
        funcs = [_get_decompose_func(tree, u) for u in tree.neighbors(node_id)]
        f = compose(tree.nodes[node_id]['f'], concatenate(*funcs))
    else:
        f = tree.nodes[0]['f']
    return f


def get_decompose_func(tree, min_size=3, max_size=30):
    f = _get_decompose_func(tree)
    decompose_func = compose(decompose_node_size(
        min_size=min_size, max_size=max_size), f)
    return decompose_func


def select_th_top(results, n_max=10, obj1_threshold=None, obj2_threshold=None):
    # remove low performance
    sel_results = [(func_tree, auc, t) for func_tree, auc,
                   t in results if auc > obj1_threshold and t < obj2_threshold]

    # make costs
    costs = np.array([(-auc, t) for func_tree, auc, t in sel_results])
    items = [func_tree for func_tree, auc, t in sel_results]

    # get Pareto shells until n_max instances
    sel_items, sel_costs = pareto_select(items, costs, n_max)

    trees = [t for ts in sel_items for t in ts]
    return trees


def select_best_trees(func_trees, evaluate_func, n_max=10, obj1_threshold=None, obj2_threshold=None, min_size=2, max_size=30, history=None):
    all_st = time.clock()
    results = []
    for func_tree in func_trees:
        st = time.clock()
        decompose_func = get_decompose_func(
            func_tree, min_size=min_size, max_size=max_size)
        auc = None
        try:
            timeout_evaluate = assign_timeout(evaluate_func, obj2_threshold)
            auc = timeout_evaluate(decompose_func)
        except:
            pass
        if auc is None:
            continue
        elapsed = time.clock() - st
        func_tree.graph['auc'] = auc
        func_tree.graph['time'] = elapsed
        func_tree.graph['id'] = 'auc:%.3f   time:%.2f' % (auc, elapsed)
        res = (func_tree.copy(), auc, elapsed)
        results.append(res)
        history[hash_tree(func_tree)] = res
    best_trees = select_th_top(results, n_max, obj1_threshold, obj2_threshold)
    all_elapsed = time.clock() - all_st
    logger.debug('Elapsed %.2f secs (%.1f mins, %.1f hours)' %
                 (all_elapsed, all_elapsed / 60, all_elapsed / 3600))
    return best_trees


def _find_comb(arr, index, num, reducedNum, store):
    if reducedNum < 0:
        return
    if reducedNum == 0:
        loc_arr = [arr[i] for i in range(index)]
        store.append(loc_arr)
        return
    prev = 1 if index == 0 else arr[index - 1]

    for k in range(prev, num + 1):
        arr[index] = k
        _find_comb(arr, index + 1, num, reducedNum - k, store)


def find_comb(n):
    store = []
    arr = [0] * n
    _find_comb(arr, 0, n, n, store)
    return store


def _combine(elements):
    if len(elements) == 1:
        for element in elements[0]:
            yield [element]
    else:
        first_elements = elements[0]
        other_elements = elements[1:]
        for element in first_elements:
            for other in _combine(other_elements):
                yield [element] + list(other)


def accept(e, elements, hash_func):
    for i in range(len(elements) - 1):
        if elements[i] == elements[i + 1]:
            if hash_func(e[i]) >= hash_func(e[i + 1]):
                return False
    return True


def each_combination(elements, hash_func):
    return [e for e in _combine(elements) if accept(e, elements, hash_func)]


def join_subtrees_to_root(g, children_sub_trees):
    sizes = [len(c) for c in children_sub_trees]
    gg = nx.disjoint_union_all([g] + children_sub_trees)
    curr_size = 1
    gg.add_edge(0, curr_size)
    for size in sizes[:-1]:
        curr_size += size
        gg.add_edge(0, curr_size)
    gg.graph = dict()
    return gg


def pre_score(tree, history):
    extract_auc = lambda res: res[1]
    get_score = lambda u: history.get(hash_tree(tree, u), [None, 0, 0])
    score = sum([extract_auc(get_score(u)) for u in tree.neighbors(0)])
    return score


def pre_select_trees(trees, n_max_sel, history):
    sel_trees = sorted(trees, key=lambda tree: pre_score(
        tree, history), reverse=True)[:n_max_sel]
    return sel_trees


def plot_pareto(history, n_max, obj1_threshold, obj2_threshold):
    palette = itertools.cycle(sns.color_palette())
    costs = np.array([(-auc, t) for func_tree, auc, t in history.values()])
    # remove low performance
    valid_results = [(func_tree, auc, t)
                     for func_tree, auc, t in history.values()
                     if auc > obj1_threshold and t < obj2_threshold]

    # make costs
    valid_costs = np.array([(-auc, t) for func_tree, auc, t in valid_results])
    valid_items = [func_tree for func_tree, auc, t in valid_results]

    sel_items, sel_costs = pareto_select(valid_items, valid_costs, n_max)
    for i, (sel_item, sel_cost) in enumerate(zip(sel_items, sel_costs)):
        print('\nshell %d' % i)
        for it, co in zip(sel_item, sel_cost):
            print('%.3f   %5.2f   %s' % (-co[0], co[1], serialize_tree(it)))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(costs[:, 0], costs[:, 1], alpha=.2)
    for i, shell_costs in enumerate(sel_costs):
        c = next(palette)
        ax.scatter(shell_costs[:, 0], shell_costs[:, 1], alpha=.8, c=[c]*shell_costs.shape[0])
        ids = np.argsort(shell_costs[:, 0])
        xx = shell_costs[ids][:, 0]
        yy = shell_costs[ids][:, 1]
        ax.step(xx, yy, where='post', c=c)
    ax.grid(linestyle=':')
    plt.show()


def generate_trees(order, decomposition_functions, evaluate_func, n_max, n_max_sel, obj1_threshold, obj2_threshold, memory, history):
    # memoize if already seen order
    if order in memory:
        return memory[order]
    # otherwise compute as:
    trees = []
    if order == 1:
        for fname in decomposition_functions:
            tree = nx.DiGraph()
            tree.add_node(0, label=fname, f=decomposition_functions[fname])
            trees.append(tree.copy())
    else:
        sizes = find_comb(order - 1)
        for children_sizes in sizes:
            # repeat for all possible node labels
            for fname in decomposition_functions:
                # generate subtrees each of the wanted size
                #   here we can use selection of possible concrete trees
                subtrees_list = [generate_trees(
                    child_size, decomposition_functions, evaluate_func, n_max,
                    n_max_sel, obj1_threshold, obj2_threshold, memory, history)
                    for child_size in children_sizes]
                # union of each subtree renumbering the nodes with current node
                for children_sub_trees in each_combination(subtrees_list, hash_tree):
                    # add edge from node to root of each child
                    g = nx.DiGraph()
                    g.add_node(0, label=fname,
                               f=decomposition_functions[fname])
                    tree = join_subtrees_to_root(g, children_sub_trees)
                    # collect all trees
                    trees.append(tree.copy())
    logger.debug('Order %d' % order)
    logger.debug('Pre-selecting up to %d among %d alternatives:' %
                 (n_max_sel, len(trees)))
    pre_sel_trees = pre_select_trees(trees, n_max_sel, history)
    logger.debug('Selecting %d among %d pre-selected alternatives:' %
                 (n_max, len(pre_sel_trees)))
    logger.debug(',  '.join([serialize_tree(t) for t in pre_sel_trees]))

    # perform learning and select with Pareto
    sel_trees = select_best_trees(
        pre_sel_trees, evaluate_func, n_max,
        obj1_threshold, obj2_threshold,
        min_size=2, max_size=30, history=history)
    logger.debug('Selected %d elements:' % len(sel_trees))
    logger.debug(',  '.join(
        ['%s %s' % (t.graph['id'], serialize_tree(t))
         for t in sel_trees]))
    memory[order] = sel_trees[:]
    plot_pareto(history, n_max, obj1_threshold, obj2_threshold)
    return sel_trees


def hyper_optimize_decomposition_function(order, decomposition_functions, evaluate_func, n_max, n_max_sel, obj1_threshold, obj2_threshold, memory, history):
    best_trees = generate_trees(
        order, decomposition_functions, evaluate_func,
        n_max, n_max_sel, obj1_threshold, obj2_threshold, memory, history)
    best_tree = max(best_trees, key=lambda t: t.graph['auc'])
    logger.debug('optimum reached at: %s   %s' %
                 (best_tree.graph['id'], serialize_tree(best_tree)))
    decompose_func = get_decompose_func(best_tree, min_size=3, max_size=30)
    return decompose_func
