#!/usr/bin/env python
"""Provides scikit interface."""
import numpy as np
import networkx as nx
from ego.decompose import do_decompose, decompose_concatenate, decompose_identity
from ego.utils.parallel_utils import parallel_map
from ego.utils.pareto_utils import pareto_select
from ego.utils.timeout_utils import assign_timeout
from ego.utils.display_utils import plot_pareto
import time
from toolz import curry
import random
import logging
logger = logging.getLogger()


def serialize_tree(tree, node_id=0):
    """serialize_tree."""
    s = tree.nodes[node_id]['label']
    n_neighbors = len(list(tree.neighbors(node_id)))
    if n_neighbors > 0:
        s = '(' + s
        for i, u in enumerate(tree.neighbors(node_id)):
            s += serialize_tree(tree, u)
            if i < n_neighbors - 1:
                s += '|'
        s += ')'
    return s


def serialize_tree_structure(tree, node_id=0):
    """serialize_tree_structure."""
    s = 'O('
    n_neighbors = len(list(tree.neighbors(node_id)))
    for i, u in enumerate(tree.neighbors(node_id)):
        s += serialize_tree(tree, u)
        if i < n_neighbors - 1:
            s += '|'
    s += ')'
    return s


def search_in_tree(tree, node_id=0, key=None, value=None):
    answer = tree.nodes[node_id][key] == value
    for i, u in enumerate(tree.neighbors(node_id)):
        answer = answer or search_in_tree(tree, u, key, value)
    return answer


def hash_tree_structure(tree, node_id=0, nbits=14):
    """hash_tree_structure."""
    bitmask = pow(2, nbits) - 1
    code = hash(serialize_tree_structure(tree, node_id)) & bitmask
    return code


def hash_tree(tree, node_id=0, nbits=14):
    """hash_tree."""
    bitmask = pow(2, nbits) - 1
    code = hash(serialize_tree(tree, node_id)) & bitmask
    return code


def get_decompose_func(tree, node_id=0):
    """get_decompose_func."""
    n_neighbors = len(list(tree.neighbors(node_id)))
    if n_neighbors > 0:
        funcs = [get_decompose_func(tree, u) for u in tree.neighbors(node_id)]
        agg_func = tree.nodes[node_id]['aggregate_function']
        cmp_func = tree.nodes[node_id]['compose_function']
        f = do_decompose(*funcs,
                         aggregate_function=agg_func,
                         compose_function=cmp_func)
    else:
        f = do_decompose(tree.nodes[node_id]['compose_function'])
    return f


def select_th_top(results, n_max=10, obj1_threshold=None, obj2_threshold=None):
    """select_th_top."""
    # remove low performance
    sel_results = [(func_tree, val1, val2) for func_tree, val1,
                   val2 in results if val1 > obj1_threshold and val2 < obj2_threshold]

    # make costs
    costs = np.array([(-val1, val2) for func_tree, val1, val2 in sel_results])
    items = [func_tree for func_tree, val1, val2 in sel_results]

    # get Pareto shells until n_max instances
    sel_items, sel_costs = pareto_select(items, costs, n_max)

    trees = [t for ts in sel_items for t in ts]
    return trees


@curry
def _evaluate_tree_(func_tree, i, evaluate_func=None, evaluate_complexity_func=None):
    decompose_func = get_decompose_func(func_tree)
    obj1 = evaluate_func(decompose_func)
    obj2 = evaluate_complexity_func(decompose_func)
    return (i, (obj1, obj2))


def select_best_trees(func_trees,
                      evaluate_func,
                      evaluate_complexity_func,
                      n_max=10,
                      obj1_threshold=None,
                      obj2_threshold=None,
                      history=None):
    """select_best_trees."""
    evaluate_tree = _evaluate_tree_(
        evaluate_func=evaluate_func,
        evaluate_complexity_func=evaluate_complexity_func)
    res_list = parallel_map(evaluate_tree, func_trees)
    results = [(func_tree.copy(), res[0], res[1])
               for res, func_tree in zip(res_list, func_trees)]
    for res in results:
        func_tree, auc, complexity = res
        func_tree.graph['auc'] = auc
        func_tree.graph['complexity'] = complexity
        history[hash_tree(func_tree)] = (func_tree, auc, complexity)
    best_trees = select_th_top(results, n_max, obj1_threshold, obj2_threshold)
    return best_trees


def _find_comb(arr, index, num, reduced_num, store):
    if reduced_num < 0:
        return
    if reduced_num == 0:
        loc_arr = [arr[i] for i in range(index)]
        store.append(loc_arr)
        return
    prev = 1 if index == 0 else arr[index - 1]

    for k in range(prev, num + 1):
        arr[index] = k
        _find_comb(arr, index + 1, num, reduced_num - k, store)


def find_comb(n):
    """find_comb."""
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
    """accept."""
    for i in range(len(elements) - 1):
        if elements[i] == elements[i + 1]:
            if hash_func(e[i]) >= hash_func(e[i + 1]):
                return False
    return True


def each_combination(elements, hash_func):
    """each_combination."""
    return [e for e in _combine(elements) if accept(e, elements, hash_func)]


def join_subtrees_to_root(g, children_sub_trees):
    """join_subtrees_to_root."""
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
    """pre_score."""
    extract_auc = lambda res: res[1]
    get_score = lambda u: history.get(hash_tree(tree, u), [None, 0, 100])
    score = np.mean([extract_auc(get_score(u)) for u in tree.neighbors(0)])
    return score


def pre_select_trees(trees, n_max_sel, history):
    """pre_select_trees."""
    sel_trees = sorted(trees, key=lambda tree: pre_score(
        tree, history), reverse=True)[:n_max_sel]
    return sel_trees


def generate_viable_candidate(children_sub_trees, compose_function, aggregate_function, decomposition_functions, evaluate_identity_func):
    cmp_func = decomposition_functions[1][compose_function]
    if aggregate_function is None:
        agg_func = decompose_concatenate
    else:
        agg_func = decomposition_functions[len(children_sub_trees)][
            aggregate_function]

    # is compose function already been used in any
    # subtree?
    is_already_present = bool(np.any([search_in_tree(
        children_sub_tree, node_id=0, key='compose_function', value=cmp_func) for children_sub_tree in children_sub_trees]))
    if is_already_present is False:
        # add edge from node to root of each child
        g_simple = nx.DiGraph()
        # simple version without compose function
        label = 'c:%s a:%s ' % ('id', aggregate_function)
        g_simple.add_node(0, label=label, aggregate_function=agg_func,
                          compose_function=decompose_identity)
        tree_simple = join_subtrees_to_root(g_simple, children_sub_trees)
        decomposition_simple = get_decompose_func(tree_simple)
        signature_simple = evaluate_identity_func(decomposition_simple)

        # candidate version with the compose function
        # add edge from node to root of each child
        g_candidate = nx.DiGraph()
        if aggregate_function is None:
            label = 'c:%s ' % (compose_function)
        else:
            label = 'c:%s a:%s ' % (compose_function, aggregate_function)
        g_candidate.add_node(
            0, label=label, aggregate_function=agg_func, compose_function=cmp_func)
        tree_candidate = join_subtrees_to_root(g_candidate, children_sub_trees)
        decomposition_candidate = get_decompose_func(tree_candidate)
        signature_candidate = evaluate_identity_func(decomposition_candidate)

        n_features = signature_candidate[0]
        if n_features > 0 and signature_simple != signature_candidate:
            return tree_candidate
    return None


@curry
def _evaluate_viable_tree_(params, i, decomposition_functions=None, evaluate_identity_func=None):
    children_sub_trees, compose_function, aggregate_function = params
    tree = generate_viable_candidate(children_sub_trees, compose_function,
                                     aggregate_function, decomposition_functions, evaluate_identity_func)
    return (i, tree)


def select_viable_trees(children_sizes, hash_tree, decomposition_functions,
                        evaluate_func, evaluate_complexity_func, evaluate_identity_func,
                        n_max, n_max_sel, n_max_sample, obj1_threshold, obj2_threshold,
                        memory, history, display, order):
    n = len(children_sizes)
    params_list = []
    counter = 0
    t0 = time.clock()
    subtrees_list = [generate_trees(
        child_size, decomposition_functions, evaluate_func, evaluate_complexity_func, evaluate_identity_func,
        n_max, n_max_sel, n_max_sample, obj1_threshold, obj2_threshold,
        memory, history, display)
        for child_size in children_sizes]
    t1 = time.clock()
    for compose_function in decomposition_functions[1]:
        if n > 1:
            for aggregate_function in decomposition_functions[n]:
                for children_sub_trees in each_combination(subtrees_list, hash_tree):
                    counter += 1
                    params_list.append((children_sub_trees, compose_function, aggregate_function))
        else:
            for children_sub_trees in each_combination(subtrees_list, hash_tree):
                counter += 1
                params_list.append((children_sub_trees, compose_function, None))
    t2 = time.clock()
    random.shuffle(params_list)
    sample_params_list = params_list[:n_max_sample]
    evaluate_viable_tree = _evaluate_viable_tree_(decomposition_functions=decomposition_functions, evaluate_identity_func=evaluate_identity_func)
    trees_list = parallel_map(evaluate_viable_tree, sample_params_list)
    viable_trees = [tree for tree in trees_list if tree is not None]
    t3 = time.clock()
    time_str = '[times for: generation of subtree lists:%.2f mins    generation of params:%.2f mins    eval viable:%.2f mins ]'%((t1-t0)/60, (t2-t1)/60, (t3-t2)/60)
    logger.debug('order:%d  #children:%d   #possible trees:%d   #viable trees:%d  in random sample of:%d   %s' % (order, n, counter, len(viable_trees), n_max_sample, time_str))
    return viable_trees


def generate_trees(order,
                   decomposition_functions,
                   evaluate_func,
                   evaluate_complexity_func,
                   evaluate_identity_func,
                   n_max,
                   n_max_sel,
                   n_max_sample,
                   obj1_threshold,
                   obj2_threshold,
                   memory,
                   history,
                   display):
    """generate_trees."""
    # memoize if already seen order
    if order in memory:
        return memory[order]
    # otherwise compute as:
    trees = []
    if order == 1:
        for fname in decomposition_functions[0]:
            tree = nx.DiGraph()
            tree.add_node(0,
                          label=fname,
                          aggregate_function=None,
                          compose_function=decomposition_functions[0][fname])
            trees.append(tree.copy())
    else:
        sizes = find_comb(order - 1)
        for children_sizes in sizes[::-1]:
            trees += select_viable_trees(
                children_sizes, hash_tree, decomposition_functions,
                evaluate_func, evaluate_complexity_func, evaluate_identity_func,
                n_max, n_max_sel, n_max_sample,
                obj1_threshold, obj2_threshold,
                memory, history, display, order)

    if len(trees) == 0:
        raise Exception('Something went wrong: no trees')
    logger.debug('Order %d' % order)
    logger.debug('Pre-selecting up to %d among %d alternatives:' %
                 (n_max_sel, len(trees)))
    pre_sel_trees = pre_select_trees(trees, n_max_sel, history)
    logger.debug('Selecting %d among %d pre-selected alternatives:' %
                 (n_max, len(pre_sel_trees)))
    logger.debug(',  '.join([serialize_tree(t) for t in pre_sel_trees]))

    # perform learning and select with Pareto
    sel_trees = select_best_trees(
        pre_sel_trees, evaluate_func, evaluate_complexity_func, n_max,
        obj1_threshold, obj2_threshold, history=history)
    logger.debug('Selected %d elements:' % len(sel_trees))
#    logger.debug(',  '.join(
#        ['%s %s' % (t.graph['id'], serialize_tree(t))
#         for t in sel_trees]))
    memory[order] = sel_trees[:]
    if display:
        plot_pareto(history, n_max, obj1_threshold, obj2_threshold)
    return sel_trees


def _hyper_optimize_decomposition_function(order,
                                           decomposition_functions,
                                           evaluate_func,
                                           evaluate_complexity_func,
                                           evaluate_identity_func,
                                           n_max,
                                           n_max_sel,
                                           n_max_sample,
                                           obj1_threshold,
                                           obj2_threshold,
                                           memory,
                                           history,
                                           display=False,
                                           return_decomposition_function_string=False):
    """hyper_optimize_decomposition_function."""
    generate_trees(
        order, decomposition_functions, evaluate_func, evaluate_complexity_func, evaluate_identity_func,
        n_max, n_max_sel, n_max_sample, obj1_threshold, obj2_threshold, memory, history, display)
    best_tree_key = max(history, key=lambda t: history[t][1])
    best_tree, best_auc, best_complexity = history[best_tree_key]
    best_tree_str = serialize_tree(best_tree)
    logger.debug('optimum reached at: %.2f   %s' % (best_auc, best_tree_str))
    decompose_func = get_decompose_func(best_tree)
    if return_decomposition_function_string is True:
        return decompose_func, best_tree_str
    else:
        return decompose_func


def hyper_optimize_decomposition_function(order,
                                          decomposition_functions,
                                          evaluate_func,
                                          evaluate_complexity_func,
                                          evaluate_identity_func,
                                          n_max,
                                          n_max_sel,
                                          n_max_sample,
                                          obj1_threshold,
                                          obj2_threshold,
                                          memory,
                                          history,
                                          display=False,
                                          return_decomposition_function_string=False,
                                          max_runtime=3600):
    """hyper_optimize_decomposition_function."""
    if max_runtime == 0:
        return _hyper_optimize_decomposition_function(
            order,
            decomposition_functions,
            evaluate_func,
            evaluate_complexity_func,
            evaluate_identity_func,
            n_max,
            n_max_sel,
            n_max_sample,
            obj1_threshold,
            obj2_threshold,
            memory,
            history,
            display,
            return_decomposition_function_string)
    else:
        timed__hyper_optimize_decomposition_function = assign_timeout(
            _hyper_optimize_decomposition_function, max_runtime)
        return timed__hyper_optimize_decomposition_function(
            order,
            decomposition_functions,
            evaluate_func,
            evaluate_complexity_func,
            evaluate_identity_func,
            n_max,
            n_max_sel,
            n_max_sample,
            obj1_threshold,
            obj2_threshold,
            memory,
            history,
            display,
            return_decomposition_function_string)
