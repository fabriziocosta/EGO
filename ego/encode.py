#!/usr/bin/env python
"""Provides scikit interface."""

from toolz import curry, compose
from collections import deque
from collections import defaultdict
import networkx as nx
from ego.component import convert
from ego.component import get_subgraphs_from_graph_component


_bitmask_ = 4294967295


def fast_hash_1(dat, bitmask=_bitmask_):
    return int(hash(dat) & bitmask) + 1


def fast_hash_2(dat_1, dat_2, bitmask=_bitmask_):
    return int(hash((dat_1, dat_2)) & bitmask) + 1


def fast_hash_3(dat_1, dat_2, dat_3, bitmask=_bitmask_):
    return int(hash((dat_1, dat_2, dat_3)) & bitmask) + 1


def fast_hash_4(dat_1, dat_2, dat_3, dat_4, bitmask=_bitmask_):
    return int(hash((dat_1, dat_2, dat_3, dat_4)) & bitmask) + 1


def fast_hash(vec, bitmask=_bitmask_):
    return int(hash(tuple(vec)) & bitmask) + 1


def fast_hash_vec(vec, bitmask=_bitmask_):
    hash_vec = []
    running_hash = 0xAAAAAAAA
    for i, vec_item in enumerate(vec):
        running_hash ^= hash((running_hash, vec_item, i))
        hash_vec.append(int(running_hash & bitmask) + 1)
    return hash_vec


def get_node_hash(g, u, seed=1):
    uh = fast_hash_2(g.nodes[u]['label'], seed)
    return uh


def get_edge_hash(g, u, v, seed=1):
    eh = fast_hash_2(g.edges[u, v]['label'], seed)
    return eh


def get_extended_edge_hash(g, u, v, seed=1):
    uh = get_node_hash(g, u, seed)
    vh = get_node_hash(g, v, seed)
    eh = get_edge_hash(g, u, v, seed)
    if uh < vh:
        edge_h = fast_hash_3(uh, vh, eh)
    else:
        edge_h = fast_hash_3(vh, uh, eh)
    return edge_h


def get_extended_node_hash(g, u, seed=1):
    uh = get_node_hash(g, u, seed)
    edges_h = [get_extended_edge_hash(g, u, v, seed) for v in g.neighbors(u)]
    nh = fast_hash(sorted(edges_h))
    ext_node_h = fast_hash_2(uh, nh)
    return ext_node_h


def rooted_breadth_first_hash(graph, root, seed=1):
    dist_list = {}
    visited = set()  # use a set as we can end up exploring few nodes
    # q is the queue containing the frontier to be expanded in the BFV
    q = deque()
    q.append(root)
    # the map associates to each vertex id the distance from the root
    dist = {}
    dist[root] = 0
    visited.add(root)
    # add vertex at distance 0
    dist_list[0] = list()
    dist_list[0].append(get_extended_node_hash(graph, root, seed))
    while len(q) > 0:
        # extract the current vertex
        u = q.popleft()
        d = dist[u] + 1
        # iterate over the neighbors of the current vertex
        for v in graph.neighbors(u):
            if v not in visited:
                dist[v] = d
                visited.add(v)
                q.append(v)
                if d not in dist_list:
                    dist_list[d] = list()
                dist_list[d].append(get_extended_node_hash(graph, v, seed))
    # hash the visit
    hash_list = []
    for d in sorted(dist_list):
        hash_list.append(fast_hash(sorted(dist_list[d])))
    hash_val = fast_hash(hash_list)
    return hash_val


def get_graph_hash(graph, seed=1):
    hashes = [rooted_breadth_first_hash(graph, u, seed) for u in graph.nodes()]
    code = fast_hash(sorted(hashes))
    return code


def get_nodes_hash(graph, seed=1):
    hashes = {u: rooted_breadth_first_hash(
        graph, u, seed) for u in graph.nodes()}
    return hashes


def edges_codes(graph, bitmask=_bitmask_, seed=1):
    edges_h = [get_edge_hash(graph, u, v, seed) &
               bitmask for u, v in graph.edges()]
    return edges_h


def vertex_codes(graph, bitmask=_bitmask_, seed=1):
    nodes_h = [get_node_hash(graph, u, seed) & bitmask for u in graph.nodes()]
    return nodes_h


def encode_graph(graph, fragmenter_func, bitmask=_bitmask_, seed=1):
    fragments = fragmenter_func(graph)
    codes = list(map(lambda g: get_graph_hash(g, seed) & bitmask, fragments))
    return codes, fragments


def get_nodes_codes(graph, fragmenter_func, bitmask=_bitmask_):
    """Compute a list of hash codes for each node."""
    codes, fragments = encode_graph(graph, fragmenter_func, bitmask)
    node_dict = defaultdict(list)
    for fragment_code, fragment in zip(codes, fragments):
        fragment_nodes_dict = get_nodes_hash(fragment)
        for u in fragment_nodes_dict:
            code = fast_hash_2(fragment_nodes_dict[u], fragment_code, bitmask)
            node_dict[u].append(code)
    return node_dict


def node_encode_graph(graph, fragmenter_func, bitmask=_bitmask_):
    encoding_codes = []
    node_ids = []
    node_codes = get_nodes_codes(graph, fragmenter_func, bitmask)
    for node_id in node_codes:
        codes = node_codes[node_id]
        node_ids += [node_id] * len(codes)
        encoding_codes += codes
    return encoding_codes, node_ids


@curry
def encode(fragments, bitmask=None, seed=1):
    codes = list(map(lambda g: get_graph_hash(g, seed) & bitmask, fragments))
    return codes, fragments


def _make_encoder(decompose_func, preprocessor=None, bitmask=None, seed=1):
    encoding_func = compose(
        encode(bitmask=bitmask, seed=seed),
        get_subgraphs_from_graph_component,
        decompose_func,
        convert)
    if preprocessor:
        encoding_func = compose(
            encoding_func,
            preprocessor)
    return encoding_func


def make_encoder(decompose_funcs, preprocessors=None, bitmask=None, seed=1):
    if isinstance(decompose_funcs, list):
        def multi_encoder(graph, bitmask=bitmask, seed=seed):
            codes = []
            fragments = []
            if not preprocessors:
                _preprocessors = [None] * len(decompose_funcs)
            else:
                _preprocessors = preprocessors
            for i, (decompose_func, preprocessor) in enumerate(
                    zip(decompose_funcs, _preprocessors)):
                encoder_func = _make_encoder(
                    decompose_func,
                    preprocessor=preprocessor,
                    bitmask=bitmask,
                    seed=seed + i)
                g = nx.Graph(graph)
                _codes, _fragments = encoder_func(g)
                codes += _codes
                fragments += _fragments
            return codes, fragments
        return multi_encoder
    else:
        return _make_encoder(
            decompose_funcs,
            preprocessor=preprocessors,
            bitmask=bitmask,
            seed=seed)
