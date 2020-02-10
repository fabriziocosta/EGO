#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from ego.utils.pareto_utils import pareto_select


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


def plot_pareto(history, n_max, obj1_threshold, obj2_threshold):
    """plot_pareto."""
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
    ax.scatter(-costs[:, 0], costs[:, 1], alpha=.2)
    for i, shell_costs in enumerate(sel_costs):
        c = next(palette)
        ax.scatter(-shell_costs[:, 0], shell_costs[:, 1],
                   alpha=.8, c=[c] * shell_costs.shape[0])
        ids = np.argsort(- shell_costs[:, 0])
        xx = - shell_costs[ids][:, 0]
        yy = shell_costs[ids][:, 1]
        ax.step(xx, yy, where='pre', c=c)
    ax.set_xlabel('AUC ROC')
    ax.set_ylabel('Complexity')
    ax.grid(linestyle=':')
    plt.show()
