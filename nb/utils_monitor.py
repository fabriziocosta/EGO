#!/usr/bin/env python
"""Provides scikit interface."""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
import time
from ego.correspondence import display_correspondences
from ego.correspondence import display_multiple_correspondences


def score2error(val):
    if val <= 1:
        return 1 - val
    else:
        return 1


def oracle_err(g, oracle_func):
    return score2error(oracle_func(g))


def score_estimator_err(g, score_estimator):
    return score2error(score_estimator.predict([g])[0])


def display_graph_list(graphs, oracle_func, score_estimator, draw_graphs=None, target_graph=None):
    true_scores = np.array([oracle_err(g, oracle_func) for g in graphs])
    oracle_ids = np.argsort(true_scores)
    pred_scores = np.array([score_estimator_err(g, score_estimator) for g in graphs])
    pred_ids = np.argsort(pred_scores)

    # oracle
    titles = []
    gs = []

    if target_graph is not None:
        titles.append('true:%.3f pred:%.3f  ' % (oracle_err(target_graph, oracle_func), score_estimator_err(target_graph, score_estimator)))
        gs.append(target_graph)

    n = 2
    for id in oracle_ids[:n]:
        true_score = true_scores[id]
        pred_score = pred_scores[id]
        titles.append('true:%.3f pred:%.3f  ' % (true_score, pred_score))
        gs.append(graphs[id])

    # preds
    for id in pred_ids[:n]:
        true_score = true_scores[id]
        pred_score = pred_scores[id]
        titles.append('true:%.3f pred:%.3f  ' % (true_score, pred_score))
        gs.append(graphs[id])

    draw_graphs(gs, titles=titles, n_graphs_per_line=2 * n + 1)

def remove_non_increasing(X,Y):
    Xout = []
    Yout = []
    x_prev = X[0] - 1
    for x,y in zip(X,Y):
        if x > x_prev:
            Xout.append(x)
            Yout.append(y)
        x_prev = x

    Xout,Yout = np.array(Xout),np.array(Yout)
    return Xout,Yout

def smooth(x_orig, y_orig, sigma=1.5):
    x,y = remove_non_increasing(x_orig,y_orig)
    xnew = np.linspace(min(x), max(x), 200)
    gy = gaussian_filter1d(y, sigma)
    f = interpolate.InterpolatedUnivariateSpline(x, gy)
    ynew = f(xnew)
    return xnew, ynew


def plot_status(estimated_mean_and_std_target, current_best, scores_list, num_oracle_queries, num_estimator_queries, sigma=1):
    fig = plt.figure(figsize=(17, 4.5))
    ax1 = fig.add_subplot(1, 1, 1)

    num_queries_to_surroagte = []
    cum = 0
    for val in num_estimator_queries:
        cum += val
        num_queries_to_surroagte.append(cum)

    x_axis = np.array(num_oracle_queries)
    if len(estimated_mean_and_std_target) == len(scores_list):
        # target with variance
        estimated_mean_and_std_target_array = np.array(estimated_mean_and_std_target)
        target_means, target_stds = estimated_mean_and_std_target_array.T

        ax1.fill_between(x_axis, target_means +
                         target_stds, target_means - target_stds, alpha=.1, color='steelblue')
        ax1.fill_between(x_axis, target_means + target_stds /
                         10, target_means - target_stds / 10, alpha=.1, color='steelblue')
        ax1.fill_between(x_axis, target_means + target_stds /
                         100, target_means - target_stds / 100, alpha=.1, color='steelblue')
        ax1.plot(x_axis, target_means, linestyle='dashed')
        xx, m = smooth(x_axis, target_means, sigma)
        ax1.plot(xx, m, lw=5, color='steelblue', label='true target graph scored by predictor')

    # median and violinplot
    #plt.violinplot(scores_list, range(len(scores_list)), points=60, widths=0.7, showmeans=True, showextrema=True, showmedians=True, bw_method=0.3)
    medians = [np.median(scores) for scores in scores_list]
    ax1.plot(x_axis, medians, color='darkorange', lw=1, linestyle='dotted')
    xx, m = smooth(x_axis, medians, sigma)
    ax1.plot(xx, m, lw=3, linestyle='dashed', color='darkorange', label='median of generated graphs scored by oracle')

    # current best
    ax1.plot(x_axis, current_best, color='darkorange', linestyle='dashed')
    xx, m = smooth(x_axis, current_best, sigma)
    ax1.plot(xx, m, lw=5, color='darkorange', label='current opt graph scored by oracle')
    ax1.legend()
    if len(estimated_mean_and_std_target) == len(scores_list):
        y_low = max(0.001, min(min(medians), min(current_best), min(target_means)))
        y_up = min(1.2, max(max(medians), max(current_best), max(target_means)))
    else:
        y_low = max(0.001, min(min(medians), min(current_best)))
        y_up = min(1.2, max(max(medians), max(current_best)))
    ax1.set_ylim(y_low, y_up)
    ax1.set_xlabel('# queries to oracle')
    ax1.set_ylabel('score')
    ax1.set_yscale('log')
    ax1.grid(axis='y')
    ax1.grid(which='minor', color='gray', alpha=.3, lw=.5)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x_axis, num_queries_to_surroagte, ':o', markersize=15, markerfacecolor='w', markeredgewidth=1, color='gray', alpha=.5, label='# queries to oracle')
    for i, (xi, yi) in enumerate(zip(x_axis, num_queries_to_surroagte)):
        plt.axvline(x=xi, c='gray', alpha=.3, lw=.5)
        ax2.text(xi, yi, str(i + 1), va='center', ha='center')
    ax2.set_ylabel('# queries to surrogate')
    fig.tight_layout()
    plt.show()


def make_monitor(target_graph=None, oracle_func=None, draw_graphs=None, draw_history=None, show_step=1, show_align=False):
    if target_graph is not None:
        target_graph = nx.convert_node_labels_to_integers(target_graph)
    history = []
    estimated_mean_and_std_target = []
    current_best = []
    scores_list = []
    duration = []
    num_oracle_queries = []
    num_estimator_queries = []

    def monitor(i, graphs, all_graphs, score_estimator, all_n_estimator_queries=None):
        if target_graph:
            assert(len(target_graph))
        num_oracle_queries.append(len(all_graphs))
        if all_n_estimator_queries is None:
            all_n_estimator_queries = 1
        num_estimator_queries.append(all_n_estimator_queries)
        history.extend(graphs[:])
        if target_graph is not None:
            mu, sigma = score_estimator.predict([target_graph]), score_estimator.predict_uncertainty([target_graph])
            estimated_mean_and_std_target.append((1 - mu[0], sigma[0]))

        true_scores = [1 - oracle_func(g) for g in graphs]
        pred_scores = 1 - score_estimator.predict(graphs)

        scores_list.append(true_scores)
        best_score = min(true_scores)
        best_graph = graphs[np.argmin(true_scores)]
        best_graph = nx.convert_node_labels_to_integers(best_graph)
        print('< %.3f > best score in %d newly generated instances' % (best_score, len(graphs)))
        score, explanation = oracle_func(best_graph, explain=True)
        explanation_str = ' '.join(['%s:%.3f' % (key, val) for key, val in sorted(explanation.items())])
        print('    score decomposition: %.3f = geom. mean of: %s' % (score, explanation_str))
        current_best.append(best_score)
        duration.append(time.process_time())
        if target_graph is not None:
            target_graph.graph['id'] = 'target'
            best_graph.graph['id'] = 'e:%.3f   s:%.3f' % (best_score, 1 - best_score)
            if show_align:
                display_correspondences(best_graph, target_graph)
        if i > 0 and (show_step == 1 or i % show_step == 0):
            if len(scores_list) >= 4:
                plot_status(estimated_mean_and_std_target, current_best, scores_list, num_oracle_queries, num_estimator_queries)
            if len(duration) > 2:
                print('Correlation coefficient true score vs predicted score: %.3f  runtime:%.1f mins' %
                      (np.corrcoef(true_scores, pred_scores)[0, 1], (duration[-1] - duration[-2]) / 60))
            if target_graph is not None:
                display_graph_list(graphs, oracle_func, score_estimator, draw_graphs=draw_graphs, target_graph=target_graph)
                if show_align:
                    print('Evolution of current best proposal w.r.t. target graph')
                    draw_evolution(best_graph, target_graph, oracle_func)
            else:
                display_graph_list(graphs, oracle_func, score_estimator, draw_graphs=draw_graphs, target_graph=None)
            print('Evolution of current best proposal')
            draw_history(graphs, oracle_func)

    return monitor


def draw_evolution(best_graph, target_graph, oracle_func):
    history = [nx.convert_node_labels_to_integers(best_graph)]
    while True:
        g = history[-1]
        parent = g.graph.get('parent', None)
        if parent is None:
            break
        parent = nx.convert_node_labels_to_integers(parent)
        history.append(parent)
    for i, g in enumerate(history):
        id = '%d) %.3f %s' % (len(history) - i, oracle_func(g), g.graph.get('type', 'orig').replace('Neighborhood', ''))
        g.graph['id'] = id

    reference = nx.convert_node_labels_to_integers(target_graph)
    display_multiple_correspondences(history, reference, n_graphs_per_line=6)
