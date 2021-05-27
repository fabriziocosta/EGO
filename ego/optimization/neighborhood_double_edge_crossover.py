#!/usr/bin/env python
"""Provides scikit interface."""

import networkx as nx
import numpy as np
import logging
import random
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


class NeighborhoodDoubleEdgeCrossover(object):

    def __init__(self, n_neighbors=None, n_double_edges=10, double_edges_fraction_selected_by_importance=0.5, part_importance_estimator=None, tournament_size_factor=10, size_n_std_to_accept=None):
        self.size_n_std_to_accept = size_n_std_to_accept
        self.tournament_size_factor = tournament_size_factor
        self.part_importance_estimator = part_importance_estimator
        self.n_neighbors = n_neighbors
        self.n_double_edges = n_double_edges
        self.double_edges_fraction_selected_by_importance = double_edges_fraction_selected_by_importance

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

    def _is_valid(self, i, j, k, l, graph):
        g = graph.copy()
        g.remove_edge(i, j)
        g.remove_edge(k, l)
        return nx.number_connected_components(g) == 2

    def _get_valid_edges(self, graph, n_double_edges=None):
        edges = list(graph.edges())
        random.shuffle(edges)
        n = len(edges)
        all_two_edges = []
        for ii in range(n - 1):
            first_edge = edges[ii]
            for jj in range(ii + 1, n):
                second_edge = edges[jj]
                if self._is_valid(first_edge[0], first_edge[1], second_edge[0], second_edge[1], graph):
                    all_two_edges.append((first_edge, second_edge))
        if len(all_two_edges) == 0:
            return None
        random.shuffle(all_two_edges)
        all_two_edges = all_two_edges[:n_double_edges]
        return all_two_edges

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

    def _score(self, i, j, k, l, graph, node_score_dict):
        g = graph.copy()
        g.remove_edge(i, j)
        g.remove_edge(k, l)
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

    def _get_selected_edges(self, graph, edges_pairs, fraction=.5):
        node_score_dict = self.part_importance_estimator.node_importance(graph)
        score_dict = {(i, j, k, l): self._score(i, j, k, l, graph, node_score_dict) for (i, j), (k, l) in edges_pairs}
        sorted_edges = sorted(edges_pairs, key=lambda e: score_dict[(e[0][0], e[0][1], e[1][0], e[1][1])], reverse=True)
        max_num = int(len(sorted_edges) * fraction)
        selected_sorted_edges = sorted_edges[:max_num]
        return selected_sorted_edges

    def _crossover(self, ga, gb):
        gs = []

        ga = nx.convert_node_labels_to_integers(ga)
        edges_a = self._get_valid_edges(ga, n_double_edges=self.n_double_edges)
        if edges_a is None:
            return gs
        edges_a = self._get_selected_edges(ga, edges_a, fraction=self.double_edges_fraction_selected_by_importance)
        if edges_a is None:
            return gs

        gb = nx.convert_node_labels_to_integers(gb)
        edges_b = self._get_valid_edges(gb, n_double_edges=self.n_double_edges)
        if edges_b is None:
            return gs
        edges_b = self._get_selected_edges(gb, edges_b, fraction=self.double_edges_fraction_selected_by_importance)
        if edges_b is None:
            return gs

        for ea in edges_a:
            for eb in edges_b:
                if ea is not None and eb is not None:
                    ia, ja = ea[0]
                    ib, jb = eb[0]
                    ibu, jbu = ib + len(ga), jb + len(ga)

                    ka, la = ea[1]
                    kb, lb = eb[1]
                    kbu, lbu = kb + len(ga), lb + len(ga)

                    gu0 = nx.disjoint_union(ga, gb)
                    self._swap_edge(gu0, (ia, ja), (ibu, jbu), mode=0)
                    self._swap_edge(gu0, (ka, la), (kbu, lbu), mode=0)
                    components = list(nx.connected_components(gu0))
                    gs.extend([nx.subgraph(gu0, component).copy() for component in components])

                    gu0 = nx.disjoint_union(ga, gb)
                    self._swap_edge(gu0, (ia, ja), (ibu, jbu), mode=0)
                    self._swap_edge(gu0, (ka, la), (kbu, lbu), mode=1)
                    components = list(nx.connected_components(gu0))
                    gs.extend([nx.subgraph(gu0, component).copy() for component in components])

                    gu1 = nx.disjoint_union(ga, gb)
                    self._swap_edge(gu1, (ia, ja), (ibu, jbu), mode=1)
                    self._swap_edge(gu1, (ka, la), (kbu, lbu), mode=1)
                    components = list(nx.connected_components(gu1))
                    gs.extend([nx.subgraph(gu1, component).copy() for component in components])

                    gu1 = nx.disjoint_union(ga, gb)
                    self._swap_edge(gu1, (ia, ja), (ibu, jbu), mode=1)
                    self._swap_edge(gu1, (ka, la), (kbu, lbu), mode=0)
                    components = list(nx.connected_components(gu1))
                    gs.extend([nx.subgraph(gu1, component).copy() for component in components])

        return gs
