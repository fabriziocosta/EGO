#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
from ego.optimization.neighborhood_graph_grammar import NeighborhoodAdaptiveGraphGrammar
from ego.optimization.neighborhood_graph_grammar import NeighborhoodPartImportanceGraphGrammar

from ego.optimization.neighborhood_edge_swap import NeighborhoodEdgeSwap
from ego.optimization.neighborhood_edge_label_swap import NeighborhoodEdgeLabelSwap
from ego.optimization.neighborhood_edge_label_mutation import NeighborhoodEdgeLabelMutation
from ego.optimization.neighborhood_edge_move import NeighborhoodEdgeMove
from ego.optimization.neighborhood_edge_remove import NeighborhoodEdgeRemove
from ego.optimization.neighborhood_edge_add import NeighborhoodEdgeAdd
from ego.optimization.neighborhood_edge_expand import NeighborhoodEdgeExpand
from ego.optimization.neighborhood_edge_contract import NeighborhoodEdgeContract

from ego.optimization.neighborhood_node_label_swap import NeighborhoodNodeLabelSwap
from ego.optimization.neighborhood_node_label_mutation import NeighborhoodNodeLabelMutation
from ego.optimization.neighborhood_node_remove import NeighborhoodNodeRemove
from ego.optimization.neighborhood_node_add import NeighborhoodNodeAdd
from ego.optimization.neighborhood_node_smooth import NeighborhoodNodeSmooth

from ego.optimization.score_estimator import GraphUpperConfidenceBoundEstimator
from ego.optimization.score_estimator import GraphRandomForestScoreEstimator
from ego.optimization.score_estimator import GraphLinearScoreEstimator
from ego.optimization.score_estimator import GraphExpectedImprovementEstimator

from ego.optimization.feasibility_estimator import FeasibilityEstimator
from ego.decomposition.paired_neighborhoods import decompose_neighborhood
from ego.vectorize import hash_graph
import time
import logging

logger = logging.getLogger()


def remove_duplicates(graphs):
    """remove_duplicates."""
    df = decompose_neighborhood(radius=2)
    selected_graphs_dict = {hash_graph(
        g, decomposition_funcs=df): g for g in graphs}
    return list(selected_graphs_dict.values())


def remove_duplicates_in_set(graphs_to_filter, graph_archive):
    """remove_duplicates_in_set."""
    df = decompose_neighborhood(radius=2)
    val_set = set([hash_graph(g, decomposition_funcs=df)
                   for g in graph_archive])
    selected_graphs = [g for g in graphs_to_filter if hash_graph(
        g, decomposition_funcs=df) not in val_set]
    return selected_graphs


def biased_sample(graphs, scores, sample_size):
    """biased_sample."""
    p = np.array(scores)
    p = np.nan_to_num(p)
    p[p < 0] = 0
    # if there are non zero probabilities
    # replace negative or zero probabilities with smallest positive prob
    if len(p[p > 0]) > 1:
        min_p = np.min(p[p > 0])
        p[p == 0] = min_p
        p = p / p.sum()
        sample_ids = np.random.choice(
            len(graphs), size=sample_size, replace=False, p=p)
    # otherwise sample uniformly at random
    else:
        sample_ids = np.random.choice(
            len(graphs), size=sample_size, replace=False)
    sample_graphs = [graphs[sample_id] for sample_id in sample_ids]
    sample_scores = [scores[sample_id] for sample_id in sample_ids]
    return sample_graphs, sample_scores


def sample(graphs, scores, sample_size, greedy_frac=0.5):
    """sample."""
    if sample_size > len(scores):
        sample_size = len(scores)

    # select with greedy strategy
    greedy_selected_graphs = []
    greedy_selected_scores = []
    n_greedy_selection = int(sample_size * greedy_frac)
    sorted_ids = np.argsort(-np.array(scores))
    greedy_selected_ids = sorted_ids[:n_greedy_selection]
    if len(greedy_selected_ids) > 0:
        greedy_selected_graphs = [graphs[greedy_selected_id]
                                  for greedy_selected_id in greedy_selected_ids]
        greedy_selected_scores = [scores[greedy_selected_id]
                                  for greedy_selected_id in greedy_selected_ids]
    # select with policy: biased sample
    policy_selected_graphs = []
    policy_selected_scores = []
    unselected_ids = sorted_ids[n_greedy_selection:]
    if len(unselected_ids) > 0:
        unselected_graphs = [graphs[unselected_id]
                             for unselected_id in unselected_ids]
        unselected_scores = [scores[unselected_id]
                             for unselected_id in unselected_ids]
        policy_selected_graphs, policy_selected_scores = biased_sample(
            unselected_graphs, unselected_scores, sample_size - len(greedy_selected_graphs))

    selected_graphs = greedy_selected_graphs + policy_selected_graphs
    selected_scores = greedy_selected_scores + policy_selected_scores
    return selected_graphs, selected_scores


def perturb(graphs, neighborhood_estimator, feasibility_estimator=None):
    """perturb."""
    # generate a fixed num of neighbors for each graph in input
    neighbor_graphs = []
    for g in graphs:
        neighbors = neighborhood_estimator.neighbors(g)
        if feasibility_estimator is None:
            feasibility_scores = [True] * len(neighbors)
        else:
            feasibility_scores = feasibility_estimator.predict(neighbors)
        for neighbor, is_feasible in zip(neighbors, feasibility_scores):
            if is_feasible:
                neighbor.graph['parent'] = g.copy()
                neighbor.graph['type'] = type(neighborhood_estimator).__name__
                neighbor_graphs.append(neighbor)
    return neighbor_graphs


def elitism(proposed_graphs, oracle_func, frac_instances_to_remove_per_iter):
    scores = [oracle_func(g) for g in proposed_graphs]
    ids = np.argsort(scores)
    n = int(frac_instances_to_remove_per_iter * len(scores))
    surviving_graphs = [proposed_graphs[id] for id in ids[n:]]
    surviving_scores = [scores[id] for id in ids[n:]]
    return surviving_graphs, surviving_scores


def optimize(graphs, oracle_func, n_iter=100,
             n_queries_to_oracle_per_iter=5, frac_instances_to_remove_per_iter=.5,
             sample_size_to_perturb=25, n_steps_driven_by_estimator=1,
             sample_size_for_grammars=None,
             neighborhood_estimators=None, score_estimator=None, feasibility_estimator=None,
             monitor=None):
    """optimize."""
    true_scores = [oracle_func(g) for g in graphs]
    proposed_graphs = []
    proposed_scores = []
    for i in range(n_iter):
        logger.info('- iteration: %d' % (i + 1))
        # update with oracle
        proposed_graphs, proposed_scores = elitism(
            proposed_graphs, oracle_func, frac_instances_to_remove_per_iter)
        graphs += proposed_graphs[:]
        true_scores += proposed_scores[:]

        # termination condition
        if max(true_scores) >= .99:
            break

        # update score_estimator
        logger.info('Fitting score estimator on %d graphs' % len(graphs))
        score_estimator.fit(graphs, true_scores)

        # update neighborhood_estimators
        if sample_size_for_grammars is None:
            neighborhood_fitting_graphs, neighborhood_fitting_scores = graphs, true_scores
        else:
            neighborhood_fitting_graphs, neighborhood_fitting_scores = sample(
                graphs,
                true_scores,
                sample_size_for_grammars,
                greedy_frac=0.5)
        logger.info('Fitting neighborhood estimators on a sample of %d graphs' %
                    len(neighborhood_fitting_graphs))

        # select small number (sample_size_to_perturb) of promising graphs for neighborhood expansion
        proposed_graphs, proposed_scores = sample(
            graphs,
            true_scores,
            sample_size_to_perturb,
            greedy_frac=0.5)

        iter_start_time = time.clock()
        logger.info('From a biased draw of %d samples from %d graphs...' % (sample_size_to_perturb, len(graphs)))
        all_proposed_graphs = []
        for n_estimator, neighborhood_estimator in enumerate(neighborhood_estimators):
            neighborhood_estimator.fit(
                neighborhood_fitting_graphs, neighborhood_fitting_scores)
            next_proposed_graphs = proposed_graphs[:]
            for step in range(n_steps_driven_by_estimator):
                start_time = time.clock()
                all_neighbor_graphs = perturb(
                    next_proposed_graphs,
                    neighborhood_estimator,
                    feasibility_estimator)
                neighbor_graphs = remove_duplicates(all_neighbor_graphs)
                neighbor_graphs = remove_duplicates_in_set(
                    neighbor_graphs, graphs)
                if len(neighbor_graphs) == 0:
                    break
                predicted_scores = score_estimator.predict(neighbor_graphs)
                # sample neighborhood according to surrogate score
                if step < n_steps_driven_by_estimator - 1:
                    sample_size = sample_size_to_perturb
                else:
                    sample_size = n_queries_to_oracle_per_iter
                next_proposed_graphs, next_proposed_scores = sample(
                    neighbor_graphs,
                    predicted_scores,
                    sample_size,
                    greedy_frac=0.5)
                end_time = time.clock()
                elapsed_time = (end_time - start_time) / 60.0
                logger.info('%d:%d/%d) %18s:%4d novel graphs (%4d generated) -> %3d selected graphs  [best predicted score:%.3f]    %.1f min' % (
                    (step + 1), n_estimator + 1, len(neighborhood_estimators),
                    type(neighborhood_estimator).__name__.replace('Neighborhood', ''),
                    len(neighbor_graphs), len(all_neighbor_graphs),
                    len(next_proposed_graphs), max(next_proposed_scores),
                    elapsed_time))
            all_proposed_graphs += next_proposed_graphs
        proposed_graphs = remove_duplicates(all_proposed_graphs)
        proposed_graphs = remove_duplicates_in_set(proposed_graphs, graphs)
        if n_queries_to_oracle_per_iter < len(proposed_graphs):
            logger.info('sampling %d out of %d total non redundant graphs generated' % (
                n_queries_to_oracle_per_iter, len(proposed_graphs)))
            predicted_scores = score_estimator.predict(proposed_graphs)
            proposed_graphs, proposed_scores = sample(
                proposed_graphs,
                predicted_scores,
                n_queries_to_oracle_per_iter,
                greedy_frac=0.5)
        if monitor:
            monitor(i, proposed_graphs, graphs, score_estimator)
        if len(proposed_graphs) == 0:
            break
        iter_end_time = time.clock()
        iter_elapsed_time_m = (iter_end_time - iter_start_time) / 60.0
        iter_elapsed_time_h = iter_elapsed_time_m / 60.0
        logger.info(' - iteration time: %.1f min (%.1f h)' % (iter_elapsed_time_m, iter_elapsed_time_h))
    return graphs


def optimizer_setup(use_RandomForest_estimator=True,
                    use_Linear_estimator=False,
                    use_EI_estimator=False,
                    use_UCB_estimator=False,
                    n_estimators=100,
                    exploration_vs_exploitation=0,
                    decomposition_score_estimator=None,

                    use_feasibility_estimator=True,
                    decomposition_feasibility_estimator=None,
                    domain_graphs_feasibility_estimator=None,

                    use_part_importance_graph_grammar=False,
                    n_neighbors_part_importance_graph_grammar=None,
                    conservativeness_part_importance_graph_grammar=1,
                    context_size_part_importance_graph_grammar=1,
                    decomposition_part_importance_graph_grammar=None,
                    domain_graphs_part_importance_graph_grammar=None,

                    use_adaptive_graph_grammar=False,
                    n_neighbors_adaptive_graph_grammar=None,
                    conservativeness_adaptive_graph_grammar=1,
                    context_size_adaptive_graph_grammar=1,
                    part_size_adaptive_graph_grammar=4,
                    decomposition_adaptive_graph_grammar_base=None,
                    decomposition_adaptive_graph_grammar_approx=None,



                    use_edge_swapping=False,
                    n_neighbors_edge_swapping=None, n_edge_swapping=1,

                    use_edge_label_swapping=False,
                    n_neighbors_edge_label_swapping=None, n_edge_label_swapping=1,

                    use_edge_label_mutation=False,
                    n_neighbors_edge_mutation=None, n_edge_mutation=1,

                    use_edge_move=False,
                    n_neighbors_edge_move=None, n_edge_move=1,

                    use_edge_removal=False,
                    n_neighbors_edge_removal=None, n_edge_removal=1,

                    use_edge_addition=False,
                    n_neighbors_edge_addition=None, n_edge_addition=1,

                    use_edge_expand=False,
                    n_neighbors_edge_expand=None, n_edge_expand=1,

                    use_edge_contract=False,
                    n_neighbors_edge_contract=None, n_edge_contract=1,


                    use_node_label_swapping=False,
                    n_neighbors_node_label_swapping=None, n_node_label_swapping=1,

                    use_node_label_mutation=False,
                    n_neighbors_node_mutation=None, n_node_mutation=1,

                    use_node_removal=False,
                    n_neighbors_node_removal=None, n_node_removal=1,

                    use_node_addition=False,
                    n_neighbors_node_addition=None, n_node_addition=1,

                    use_node_smooth=False,
                    n_neighbors_node_smooth=None, n_node_smooth=1):
    """optimizer_setup."""
    neighborhood_estimators = []

    if use_edge_swapping:
        nes = NeighborhoodEdgeSwap(
            n_edges=n_edge_swapping, n_neighbors=n_neighbors_edge_swapping)
        neighborhood_estimators.append(nes)

    if use_edge_label_swapping:
        nels = NeighborhoodEdgeLabelSwap(
            n_edges=n_edge_label_swapping, n_neighbors=n_neighbors_edge_label_swapping)
        neighborhood_estimators.append(nels)

    if use_edge_label_mutation:
        nelm = NeighborhoodEdgeLabelMutation(
            n_edges=n_edge_mutation, n_neighbors=n_neighbors_edge_mutation)
        neighborhood_estimators.append(nelm)

    if use_edge_move:
        nem = NeighborhoodEdgeMove(
            n_edges=n_edge_move, n_neighbors=n_neighbors_edge_move)
        neighborhood_estimators.append(nem)

    if use_edge_removal:
        ner = NeighborhoodEdgeRemove(
            n_edges=n_edge_removal, n_neighbors=n_neighbors_edge_removal)
        neighborhood_estimators.append(ner)

    if use_edge_addition:
        nea = NeighborhoodEdgeAdd(
            n_edges=n_edge_addition, n_neighbors=n_neighbors_edge_addition)
        neighborhood_estimators.append(nea)

    if use_edge_expand:
        nee = NeighborhoodEdgeExpand(
            n_edges=n_edge_expand, n_neighbors=n_neighbors_edge_expand)
        neighborhood_estimators.append(nee)

    if use_edge_contract:
        nec = NeighborhoodEdgeContract(
            n_edges=n_edge_contract, n_neighbors=n_neighbors_edge_contract)
        neighborhood_estimators.append(nec)

    if use_node_label_swapping:
        nnls = NeighborhoodNodeLabelSwap(
            n_nodes=n_node_label_swapping, n_neighbors=n_neighbors_node_label_swapping)
        neighborhood_estimators.append(nnls)

    if use_node_label_mutation:
        nnlm = NeighborhoodNodeLabelMutation(
            n_nodes=n_node_mutation, n_neighbors=n_neighbors_node_mutation)
        neighborhood_estimators.append(nnlm)

    if use_node_removal:
        nnr = NeighborhoodNodeRemove(
            n_nodes=n_node_removal, n_neighbors=n_neighbors_node_removal)
        neighborhood_estimators.append(nnr)

    if use_node_addition:
        nna = NeighborhoodNodeAdd(
            n_nodes=n_node_addition, n_neighbors=n_neighbors_node_addition)
        neighborhood_estimators.append(nna)

    if use_node_smooth:
        nns = NeighborhoodNodeSmooth(
            n_nodes=n_node_smooth, n_neighbors=n_neighbors_node_smooth)
        neighborhood_estimators.append(nns)

    if use_part_importance_graph_grammar:
        npigge = NeighborhoodPartImportanceGraphGrammar(
            decomposition_function=decomposition_part_importance_graph_grammar,
            context=context_size_part_importance_graph_grammar,
            count=conservativeness_part_importance_graph_grammar,
            frac_nodes_to_select=.5,
            n_neighbors=n_neighbors_part_importance_graph_grammar)
        npigge.fit_grammar(domain_graphs_part_importance_graph_grammar)
        neighborhood_estimators.append(npigge)
        logger.info('Part Importance Graph Grammar: %s' % npigge)

    if use_adaptive_graph_grammar:
        ane = NeighborhoodAdaptiveGraphGrammar(
            base_decomposition_function=decomposition_adaptive_graph_grammar_base,
            approximate_decomposition_function=decomposition_adaptive_graph_grammar_approx,
            context=context_size_adaptive_graph_grammar,
            count=conservativeness_adaptive_graph_grammar,
            n_neighbors=n_neighbors_adaptive_graph_grammar,
            ktop=part_size_adaptive_graph_grammar,
            enforce_connected=True)
        neighborhood_estimators.append(ane)

    if use_UCB_estimator:
        score_estimator = GraphUpperConfidenceBoundEstimator(
            decomposition_funcs=decomposition_score_estimator,
            exploration_vs_exploitation=exploration_vs_exploitation)
    if use_RandomForest_estimator:
        score_estimator = GraphRandomForestScoreEstimator(
            decomposition_funcs=decomposition_score_estimator,
            n_estimators=n_estimators,
            exploration_vs_exploitation=exploration_vs_exploitation)
    if use_Linear_estimator:
        score_estimator = GraphLinearScoreEstimator(
            decomposition_funcs=decomposition_score_estimator,
            n_estimators=n_estimators,
            exploration_vs_exploitation=exploration_vs_exploitation)
    if use_EI_estimator:
        score_estimator = GraphExpectedImprovementEstimator(
            decomposition_funcs=decomposition_score_estimator,
            exploration_vs_exploitation=exploration_vs_exploitation)

    if use_feasibility_estimator:
        feasibility_estimator = FeasibilityEstimator(
            decomposition_funcs=decomposition_feasibility_estimator)
        logger.info('Fitting feasibility estimator on %d graphs' % len(domain_graphs_feasibility_estimator))
        feasibility_estimator.fit(domain_graphs_feasibility_estimator)
    else:
        feasibility_estimator = None
    return neighborhood_estimators, score_estimator, feasibility_estimator
