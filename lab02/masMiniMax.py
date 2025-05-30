# masMiniMax.py - Minimax search with alpha-beta pruning
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

def minimax(node,depth):
    """returns the value of node, and a best path for the agents
    """
    if node.isLeaf():
        return node.evaluate(),None
    elif node.isMax:
        max_score = float("-inf")
        max_path = None
        for C in node.children():
            score,path = minimax(C,depth+1)
            if score > max_score:
                max_score = score
                max_path = C.name,path
        return max_score,max_path
    else:
        min_score = float("inf")
        min_path = None
        for C in node.children():
            score,path = minimax(C,depth+1)
            if score < min_score:
                min_score = score
                min_path = C.name,path
        return min_score,min_path

def minimax_alpha_beta(node, alpha, beta, depth=0):
    """node is a Node, 
       alpha and beta are cutoffs
       depth is the depth on node (for indentation in printing)
    returns value, path
    where path is a sequence of nodes that results in the value
    """
    node.display(2,"  "*depth, f"minimax_alpha_beta({node.name}, {alpha}, {beta})")
    best=None      # only used if it will be pruned
    if node.isLeaf():
        node.display(2,"  "*depth, f"{node} leaf value {node.evaluate()}")
        return node.evaluate(),None
    elif node.isMax:
        for C in node.children():
            score,path = minimax_alpha_beta(C,alpha,beta,depth+1)
            if score >= beta:    # beta pruning
                node.display(2,"  "*depth, f"{node} pruned {beta=}, {C=}")
                return score, None 
            if score > alpha:
                alpha = score
                best = C.name, path
        node.display(2,"  "*depth, f"{node} returning max {alpha=}, {best=}")
        return alpha,best
    else:
        for C in node.children():
            score,path = minimax_alpha_beta(C,alpha,beta,depth+1)
            if score <= alpha:     # alpha pruning
                node.display(2,"  "*depth, f"{node} pruned {alpha=}, {C=}")
                return score, None
            if score < beta:
                beta=score
                best = C.name,path
        node.display(2,"  "*depth, f"{node} returning min {beta=}, {best=}")
        return beta,best

from masProblem import fig10_5, Magic_sum, Node

# Node.max_display_level=2   # print detailed trace
# minimax_alpha_beta(fig10_5, -9999, 9999,0)
# minimax_alpha_beta(Magic_sum(), -9999, 9999,0)

#To test much time alpha-beta pruning can save over minimax:
## import timeit
## timeit.Timer("minimax(Magic_sum(),0)",setup="from __main__ import minimax, Magic_sum").timeit(number=1)
## timeit.Timer("minimax_alpha_beta(Magic_sum(), -9999, 9999,0)", setup="from __main__ import minimax_alpha_beta, Magic_sum").timeit(number=1)

