#!/usr/bin/env python
"""Provides scikit interface."""


def decompose_identity(graph_component):
    return graph_component

def idn(*args, **kargs): 
    return decompose_identity(*args, **kargs)