#!/usr/bin/env python
"""Provides scikit interface."""

import networkx as nx
import numpy as np
import random
import logging
logger = logging.getLogger(__name__)


def make_size_filter(graphs, z=2):
    lens = [len(g) for g in graphs]
    mu, sigma = np.mean(lens), np.std(lens)

    def size_filter(g):
        if z is None:
            return True
        if mu - z * sigma < len(g) < mu + z * sigma:
            return True
        else:
            return False
    return size_filter


def filter_by_size(graphs, size_filter=None):
    return [g for g in graphs if size_filter(g)]


class NeighborhoodEdgeCrossover(object):

    def __init__(self, n_edges=1, n_neighbors=None, part_importance_estimator=None, tournament_size_factor=10, size_n_std_to_accept=None):
        self.size_n_std_to_accept = size_n_std_to_accept
        self.tournament_size_factor = tournament_size_factor
        self.part_importance_estimator = part_importance_estimator
        self.n_edges = n_edges
        self.n_neighbors = n_neighbors

    def fit(self, graphs, targets=None):
        """fit."""
        self.size_filter = make_size_filter(graphs, z=self.size_n_std_to_accept)
        if targets is None:
            self.graphs = graphs[:]
        else:
            # store graphs sorted by target
            ids = sorted(range(len(graphs)), key=lambda i: targets[i], reverse=True)
            self.graphs = [graphs[id].copy() for id in ids]
        return self

    def neighbors(self, graph):
        """neighbors."""
        if self.n_neighbors is None:
            n = len(self.graphs)
        else:
            n = min(len(self.graphs), self.n_neighbors)
        neighs = []
        m = self.tournament_size_factor * n
        m = min(m, len(self.graphs))
        # select the best n out of m random graphs
        # sample m integers from 0 to len(graphs)
        ids = random.sample(range(len(self.graphs)), m)
        # sort them and select the smallest n
        sorted_ids = sorted(ids)[:n]
        for i in sorted_ids:
            candidate_graphs = self._crossover(graph, self.graphs[i])
            neighs.extend(filter_by_size(candidate_graphs, size_filter=self.size_filter))
        return neighs

    def _is_valid(self, i, j, graph):
        g = graph.copy()
        g.remove_edge(i, j)
        return nx.number_connected_components(g) == 2

    def _get_valid_edges_based_on_centrality(self, graph, max_num=None):
        centrality_dict = nx.edge_betweenness_centrality(graph)
        edges = centrality_dict.keys()
        sorted_edges = sorted(edges, key=lambda e: centrality_dict[(e[0], e[1])], reverse=True)
        valid_edges = []
        counter = 0
        for i, j in sorted_edges:
            if self._is_valid(i, j, graph):
                valid_edges.append((i, j))
                counter += 1
                if max_num is not None and counter >= max_num:
                    break
        return valid_edges

    def _score(self, i, j, graph, node_score_dict):
        g = graph.copy()
        g.remove_edge(i, j)
        connected_components = list(nx.connected_components(g))
        if len(connected_components) != 2:
            return 0
        vals_0 = [node_score_dict[u] for u in connected_components[0]]
        mu_0, sigma_0 = np.mean(vals_0), np.std(vals_0)
        sq_err_0 = (sigma_0 * sigma_0) / float(len(vals_0))
        vals_1 = [node_score_dict[u] for u in connected_components[1]]
        mu_1, sigma_1 = np.mean(vals_1), np.std(vals_1)
        sq_err_1 = (sigma_1 * sigma_1) / float(len(vals_1))
        if sigma_0 == 0 or sigma_1 == 0:
            return 0
        # compute z-statistics
        score_value = np.abs(mu_0 - mu_1) / np.sqrt(sq_err_0 + sq_err_1)
        return score_value

    def _get_valid_edges_based_on_part_importance(self, graph, max_num=None):
        node_score_dict = self.part_importance_estimator.node_importance(graph)
        score_dict = {(i, j): self._score(i, j, graph, node_score_dict) for i, j in graph.edges()}
        edges = list(graph.edges())
        sorted_edges = sorted(edges, key=lambda e: score_dict[(e[0], e[1])], reverse=True)
        selected_sorted_edges = sorted_edges[:max_num]
        return selected_sorted_edges

    def _get_valid_edges(self, graph, max_num=None):
        if max_num is None:
            valid_edges = [(i,j) for i, j in graph.edges() if self._is_valid(i,j,graph)]
            return valid_edges
        if self.part_importance_estimator is None:
            return self._get_valid_edges_based_on_centrality(graph, max_num)
        else:
            return self._get_valid_edges_based_on_part_importance(graph, max_num)

    def _swap_edge(self, g, e1, e2, mode=0):
        s1, d1 = e1
        s2, d2 = e2
        if mode == 0:
            ne1 = (s1, d2)
            ne2 = (s2, d1)
        else:
            ne1 = (s1, s2)
            ne2 = (d2, d1)
        args_e1 = g.edges[e1[0], e1[1]]
        args_e2 = g.edges[e2[0], e2[1]]
        g.remove_edge(*e1)
        g.remove_edge(*e2)
        g.add_edge(*ne1, **args_e1)
        g.add_edge(*ne2, **args_e2)

    def _crossover(self, ga, gb):
        gs = []

        ga = nx.convert_node_labels_to_integers(ga)
        gb = nx.convert_node_labels_to_integers(gb)

        edges_a = self._get_valid_edges(ga, max_num=self.n_edges)
        edges_b = self._get_valid_edges(gb, max_num=self.n_edges)
        for ea in edges_a:
            for eb in edges_b:
                if ea is not None and eb is not None:
                    ia, ja = ea
                    ib, jb = eb
                    ibu, jbu = ib + len(ga), jb + len(ga)

                    gu0 = nx.disjoint_union(ga, gb)
                    self._swap_edge(gu0, (ia, ja), (ibu, jbu), mode=0)
                    components = list(nx.connected_components(gu0))
                    gs.extend([nx.subgraph(gu0, component).copy() for component in components])

                    gu1 = nx.disjoint_union(ga, gb)
                    self._swap_edge(gu1, (ia, ja), (ibu, jbu), mode=1)
                    components = list(nx.connected_components(gu1))
                    gs.extend([nx.subgraph(gu1, component).copy() for component in components])
        return gs
