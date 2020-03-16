#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import networkx as nx
import random
from ego.decompose import do_decompose
from ego.decomposition.positive_and_negative import decompose_positive, decompose_negative
from ego.decomposition.union import decompose_all_union
from graphlearn.sample import LocalSubstitutionGraphGrammarSample as lsgg
from graphlearn.lsgg_ego import lsgg_ego
from ego.optimization.part_importance_estimator import PartImportanceEstimator
import logging
logger = logging.getLogger(__name__)


class NeighborhoodGraphGrammar(object):

    def __init__(self,
                 core=2,
                 context=1,
                 count=2,
                 n_neighbors=None,
                 perturbation_size=2,
                 part_importance_estimator=None,
                 objective_func=None):
        self.n_neighbors = n_neighbors
        self.graph_grammar = lsgg(
            radii=list(range(core)),
            thickness=context,
            filter_min_cip=count,
            filter_min_interface=2,
            nodelevel_radius_and_thickness=False)
        self.perturbation_size = perturbation_size
        self.part_importance_estimator = part_importance_estimator
        self.objective_func = objective_func

    def fit(self, graphs):
        self.graph_grammar.fit(graphs)
        return self

    def __repr__(self):
        return str(self.graph_grammar)

    def perturb(self, orig_graph):
        graph = nx.Graph(orig_graph)
        for i in range(self.perturbation_size):
            # select a node at random
            ns = [u for u in graph.nodes()]
            root = np.random.choice(ns, size=1)[0]
            # materialize the neighborhood
            graphs_it = self.graph_grammar.root_neighbors(
                graph, [root], n_neighbors=self.n_neighbors)
            graphs = list(graphs_it) + [graph]
            # select a neighbor at random
            ids = list(range(len(graphs)))
            id = np.random.choice(ids, size=1)[0]
            graph = nx.Graph(graphs[id])
        return graph

    def _select_nodes_in_prob(self, graph, node_score_dict, edge_score_dict):
        # select root nodes with probability proportional to negative scores
        _eps_ = 1e-4
        s = np.array([node_score_dict[u] + _eps_ for u in graph.nodes()])
        # invert score and take exp
        p = np.exp(-s / np.max(np.absolute(s)))
        # normalize it to make it a prob
        p = p / p.sum()
        ns = [u for u in graph.nodes()]
        # sample up to half of the most negative nodes
        n = int(len(graph) / 2)
        sel = np.random.choice(ns, size=n, p=p, replace=False)
        selected_nodes = list(sel)
        return selected_nodes

    def _select_negative_nodes(self, graph, node_score_dict, edge_score_dict):
        selected_nodes = [u for u in graph.nodes() if node_score_dict[u] <= 0]
        return selected_nodes

    def _select_best(self, graphs):
        # select graph that maximizes the average objective score
        # (to contrast the tendency to produce large molecules)
        func = lambda g: self.objective_func(g) / float(len(g))
        best_graph = max(graphs, key=func)
        return best_graph

    def neighbors(self, graph):
        if self.n_neighbors is None:
            return list(self.graph_grammar.neighbors(graph))
        else:
            return list(self.graph_grammar.neighbors_sample(graph, self.n_neighbors))

    def make_neighbors(self, graph, selected_nodes=None, include_original=True):
        if selected_nodes is None or len(selected_nodes) == 0:
            out_graphs = self.make_all_neighbors(graph, include_original)
        else:
            out_graphs = self.make_neighbors_from_selected_nodes(
                graph, selected_nodes, include_original)
        return out_graphs

    def make_all_neighbors(self, graph, include_original=True):
        if include_original:
            out_graphs = [graph]
        else:
            out_graphs = []
        out_graphs += self.neighbors(graph)
        return out_graphs

    def make_neighbors_from_selected_nodes(self, graph, selected_nodes=None, include_original=True):
        # compute neighborhood, i.e. graphs that are obtained as a single core
        # substitution
        if include_original:
            out_graphs = [graph]
        else:
            out_graphs = []
        for selected_node in selected_nodes:
            graphs_it = self.graph_grammar.root_neighbors(
                graph, [selected_node], n_neighbors=self.n_neighbors)
            out_graphs += list(graphs_it)
        return out_graphs

    def make_gradient_neighbors(self, graph):
        if self.part_importance_estimator is None:
            selected_nodes = None
        else:
            res = self.part_importance_estimator.predict(graph)
            node_score_dict, edge_score_dict = res

            selected_nodes = self._select_negative_nodes(
                graph, node_score_dict, edge_score_dict)

        out_graphs = self.make_neighbors(graph, selected_nodes)
        return out_graphs

    def gradient_descent(self, graph):
        out_graphs = self.make_gradient_neighbors(graph)
        best_graph = self._select_best(out_graphs)
        return best_graph


# ----------------------------------------------------------

class NeighborhoodEgoGraphGrammar(object):

    def __init__(self,
                 decomposition_function=None,
                 context=1,
                 count=1,
                 n_neighbors=None,
                 perturbation_size=0,
                 objective_func=None):
        self.n_neighbors = n_neighbors
        self.graph_grammar = lsgg_ego(
            decomposition_function=decomposition_function,
            thickness=context,
            filter_min_cip=count,
            filter_min_interface=2,
            nodelevel_radius_and_thickness=False)
        self.perturbation_size = perturbation_size
        self.objective_func = objective_func

    def fit(self, graphs):
        self.graph_grammar.fit(graphs)
        return self

    def __repr__(self):
        return str(self.graph_grammar)

    def neighbors(self, graph, n_neighbors=None):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if n_neighbors is None:
            ns = list(self.graph_grammar.neighbors(graph))
        else:
            ns = list(self.graph_grammar.neighbors_sample(graph, n_neighbors))
        return ns

    def perturb(self, orig_graph):
        graph = nx.Graph(orig_graph)
        for i in range(self.perturbation_size):
            # select a graph at random
            out_graphs = self.neighbors(graph)
            if len(out_graphs) > 2:
                graph = np.random.choice(out_graphs, size=1)[0]
        return graph

    def _select_best(self, graphs):
        # select graph that maximizes the average objective score
        # (to contrast the tendency to produce large molecules)
        func = lambda g: self.objective_func(g) / float(len(g))
        best_graph = max(graphs, key=func)
        return best_graph

    def gradient_descent(self, graph):
        out_graphs = self.neighbors(graph)
        best_graph = self._select_best([graph] + out_graphs)
        return best_graph


# ----------------------------------------------------------

class NeighborhoodPartImportanceGraphGrammar(object):

    def __init__(self,
                 decomposition_function=None,
                 context=1,
                 count=1,
                 n_neighbors=None,
                 frac_nodes_to_select=.5,
                 enforce_connected=True):
        self.decomposition_function = decomposition_function
        self.n_neighbors = n_neighbors
        self.part_importance_estimator = PartImportanceEstimator(
            decompose_func=decomposition_function)
        self.graph_grammar = lsgg_ego(
            decomposition_function=decomposition_function,
            thickness=context,
            filter_min_cip=count,
            filter_min_interface=2,
            nodelevel_radius_and_thickness=False)
        self.frac_nodes_to_select = frac_nodes_to_select
        self.enforce_connected = enforce_connected

    def fit_grammar(self, graphs):
        self.graph_grammar.fit(graphs)
        return self

    def fit_part_importance_estimator(self, graphs, targets):
        self.part_importance_estimator.fit(graphs, targets)
        return self

    def fit(self, graphs, targets):
        return self.fit_part_importance_estimator(graphs, targets)

    def __repr__(self):
        return str(self.graph_grammar)

    def neighbors(self, graph, n_neighbors=None):
        res = self.part_importance_estimator.predict(graph)
        node_score_dict, edge_score_dict = res
        nodes = list(graph.nodes())
        selected_nodes = sorted(nodes, key=lambda u: node_score_dict[u])
        selected_nodes = selected_nodes[
            :int(len(selected_nodes) * self.frac_nodes_to_select)]
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if n_neighbors is None:
            ns = list(self.graph_grammar.neighbors(graph))
        else:
            ns = list(self.graph_grammar.root_neighbors(
                graph, selected_nodes, n_neighbors))
            random.shuffle(ns)
            ns = ns[:n_neighbors]
        if self.enforce_connected:
            ns = [g for g in ns if nx.is_connected(g)]
        return ns


# ----------------------------------------------------------

class NeighborhoodAdaptiveGraphGrammar(object):

    def __init__(self,
                 base_decomposition_function=None,
                 approximate_decomposition_function=None,
                 context=1,
                 count=1,
                 n_neighbors=None,
                 ktop=4,
                 enforce_connected=True):
        self.ktop = ktop
        self.base_decomposition_function = base_decomposition_function
        self.approximate_decomposition_function = approximate_decomposition_function
        self.n_neighbors = n_neighbors
        self.part_importance_estimator = PartImportanceEstimator(
            decompose_func=self.base_decomposition_function)
        self.graph_grammar = lsgg_ego(
            decomposition_function=self.base_decomposition_function,
            thickness=context,
            filter_min_cip=count,
            filter_min_interface=2,
            nodelevel_radius_and_thickness=False)
        self.enforce_connected = enforce_connected

    def fit(self, graphs, targets):
        self.fit_part_importance_estimator(graphs, targets)

        pos_dec = do_decompose(decompose_positive(ktop=self.ktop, part_importance_estimator=self.part_importance_estimator),
                               compose_function=decompose_all_union)
        neg_dec = do_decompose(decompose_negative(ktop=self.ktop, part_importance_estimator=self.part_importance_estimator),
                               compose_function=decompose_all_union)
        frag_dec = do_decompose(pos_dec, neg_dec, compose_function=self.approximate_decomposition_function)
        self.adaptive_decomposition_function = do_decompose(pos_dec, neg_dec, frag_dec)
        self.graph_grammar.set_decomposition(self.adaptive_decomposition_function)
        self.fit_grammar(graphs)
        return self

    def fit_grammar(self, graphs):
        self.graph_grammar.fit(graphs)
        return self

    def fit_part_importance_estimator(self, graphs, targets):
        self.part_importance_estimator.fit(graphs, targets)
        return self

    def __repr__(self):
        return str(self.graph_grammar)

    def neighbors(self, graph, n_neighbors=None):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if n_neighbors is None:
            ns = list(self.graph_grammar.neighbors(graph))
        else:
            ns = list(self.graph_grammar.neighbors_sample(graph, n_neighbors))
        if self.enforce_connected:
            ns = [g for g in ns if nx.is_connected(g)]
        return ns
