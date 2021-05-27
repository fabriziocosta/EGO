#!/usr/bin/env python
"""Provides scikit interface."""

from ego.component import GraphComponent
from collections import Counter


def decompose_relabel_node_size(graph_component):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        node_size_label = '%d' % subgraph.number_of_nodes()
        new_signature = signature + '_node_size_' + node_size_label
        new_subgraphs_list.append(subgraph)
        new_signatures_list.append(new_signature)
    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


def decompose_relabel_node_label_frequency(graph_component):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        labels = [subgraph.nodes[u]['label'] for u in subgraph.nodes()]
        new_signature = signature + '_node_label_frequency_' + '_'.join(['%s:%s' % (k, v) for k, v in Counter(labels).most_common()])
        new_subgraphs_list.append(subgraph)
        new_signatures_list.append(new_signature)
    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


def decompose_relabel_distinct_node_labels(graph_component):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        labels = [subgraph.nodes[u]['label'] for u in subgraph.nodes()]
        new_signature = signature + '_distinct_node_labels_%d' % len(set(labels))
        new_subgraphs_list.append(subgraph)
        new_signatures_list.append(new_signature)
    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


def decompose_relabel_node_degree_frequency(graph_component):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        labels = [v for v in dict(subgraph.degree()).values()]
        new_signature = signature + '_node_degree_frequency_' + '_'.join(['%s:%s' % (k, v) for k, v in Counter(labels).most_common()])
        new_subgraphs_list.append(subgraph)
        new_signatures_list.append(new_signature)
    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


def decompose_relabel_max_node_degree(graph_component):
    new_subgraphs_list = []
    new_signatures_list = []
    for subgraph, signature in zip(graph_component.subgraphs, graph_component.signatures):
        labels = [v for v in dict(subgraph.degree()).values()]
        new_signature = signature + '_max_node_degree_%d' % max(labels)
        new_subgraphs_list.append(subgraph)
        new_signatures_list.append(new_signature)
    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


def decompose_relabel_estimator(graph_component, graph_estimator=None):
    new_subgraphs_list = []
    new_signatures_list = []
    preds = graph_estimator.predict(graph_component.subgraphs)
    for subgraph, signature, pred in zip(graph_component.subgraphs, graph_component.signatures, preds):
        new_signature = signature + '_estimator_%s' % pred
        new_subgraphs_list.append(subgraph)
        new_signatures_list.append(new_signature)
    gc = GraphComponent(
        graph=graph_component.graph,
        subgraphs=new_subgraphs_list,
        signatures=new_signatures_list)
    return gc


def rlbest(*args, **kargs): 
    return decompose_relabel_estimator(*args, **kargs)

def rlbmdgr(*args, **kargs): 
    return decompose_relabel_max_node_degree(*args, **kargs)

def rlbdfrq(*args, **kargs): 
    return decompose_relabel_node_degree_frequency(*args, **kargs)

def rlbnod(*args, **kargs): 
    return decompose_relabel_distinct_node_labels(*args, **kargs)

def rlblfrq(*args, **kargs): 
    return decompose_relabel_node_label_frequency(*args, **kargs)

def rlbsiz(*args, **kargs): 
    return decompose_relabel_node_size(*args, **kargs)


