#!/usr/bin/env python3

#  (C) Copyright Wieger Wesselink 2022. Distributed under the GPL-3.0
#  Software License, (See accompanying file license.txt or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

from draughts1 import *

import math
import io
import random
import copy
from typing import List

Move = int  # moves are stored as integers

def init_scan():
    Scan.set("variant", "normal")
    Scan.set("book", "false")
    Scan.set("book-ply", "4")
    Scan.set("book-margin", "4")
    Scan.set("ponder", "false")
    Scan.set("threads", "1")
    Scan.set("tt-size", "24")
    Scan.set("bb-size", "0")
    Scan.update()
    Scan.init()


class MCTSNode(object):
    index = 0

    def __init__(self, state: Pos):
        self.state = state
        self.moves = generate_moves(state)
        self.children = []
        self.Q = 0  # the total score of the simulations through this node
        self.N = 0  # the number of simulations through this node
        self.expanded_move_count = 0  # the number of moves in self.moves that have been expanded
        self.parent = None # the parent node

    def is_fully_expanded(self) -> bool:
        return self.expanded_move_count == len(self.moves)


class MCTSTree(object):
    def __init__(self, state: Pos):
        self.nodes = [MCTSNode(state)]

    def root(self):
        return self.nodes[0]

    # Adds a child v to node u, such that play_move(u->state, u->moves[i]) == v->state.
    # Modifies u (N.B. the order of the moves may be changed).
    def add_child(self, u: MCTSNode, i: int) -> MCTSNode:
        j = u.expanded_move_count
        if i > j:
            u.moves.swap(i, j)  # swap moves i and j
        v = MCTSNode(u.state.succ(u.moves[j]))
        u.children.append(v)
        u.expanded_move_count += 1
        v.parent = u
        return v


def find_move(u: Pos, v: Pos) -> Move:
    moves = generate_moves(u)
    for m in moves:
        if u.succ(m) == v:
            return m
    return move_none()


def print_move_between_positions(src: Pos, dest: Pos) -> str:
    m = find_move(src, dest)
    return print_move(m, src) if m else f'jump({print_position(dest, False, True)})'


def print_path(path: List[MCTSNode]) -> str:
    out = io.StringIO()
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]
        out.write(print_move_between_positions(u.state, v.state) + ' ')
    return out.getvalue()


def opponent_score(u: MCTSNode) -> float:
    return u.N - u.Q if u.state.is_white_to_move() else u.Q


def uct(u: MCTSNode, v: MCTSNode, c: float = 0):
    return opponent_score(v) / v.N + c * math.sqrt(2 * math.log(u.N) / v.N)


def best_child(u: MCTSNode, c: float) -> MCTSNode:
    assert u.children

    def key(i: int) -> float:
        v = u.children[i]
        return uct(u, v, c)

    i = max(range(len(u.children)), key=key)
    return u.children[i]


# print the uct scores of the root of the tree
def log_uct_scores(tree: MCTSTree, c: float) -> None:
    print('uct scores')
    u = tree.root()
    for v in u.children:
        print(f'{print_move_between_positions(u.state, v.state)} uct = {uct(u, v, c):.4f} N = {v.N}')
    print('')


def expand(tree: MCTSTree, u: MCTSNode) -> MCTSNode:
    i = random.randrange(u.expanded_move_count, len(u.moves))  # u.expanded_move_count <= i < len(u.moves)
    return tree.add_child(u, i)


def normalize(x: float) -> float:
    if x > 0:
        return 1
    elif x < 0:
        return 0
    else:
        return 0.5


# use a piece count evaluation and normalize it to values in the interval [0,1]
def simulate(u: MCTSNode) -> float:
    # value = piece_count_eval(play_forced_moves(u.state))
    value = playout_random(play_forced_moves(u.state), 150, False)
    return normalize(value)


# add a connection from the root to the end of the path
def add_root_connection(tree: MCTSTree, path: List[MCTSNode]) -> None:
    u = tree.root()
    v = path[-1]
    if u.state.is_white_to_move() == v.state.is_white_to_move():
        v = path[-2]
    u.children.append(v)

# add a connection from the root child to the end of the path
def add_enemy_root_connection(u: MCTSNode, path: List[MCTSNode]) -> None:
    v = path[-1]
    if u.state.is_white_to_move() == v.state.is_white_to_move():
        v = path[-2]
    u.children.append(v)

def mcts_traps(tree: MCTSTree, c: float, max_iterations, useFPN=True, useEnemyFPN=False, verbose=False) -> MCTSNode:
    for i in range(max_iterations):
        if verbose:
            if i % 1000 == 0 and i > 0:
                print(f'i = {i}')

        u = tree.root()
        path = [u]
        forced = True
        enemyForced = True
        rootChild = None

        # selection of a leaf node
        while u.moves:
            if len(path) % 2 == 0 and len(u.moves) > 1:
                forced = False

            if len(path) == 2:
                rootChild = copy.copy(u)
            if len(path) % 2 == 1 and len(path) > 1 and len(u.moves) > 1:
                enemyForced = False

            if u.is_fully_expanded():
                u = best_child(u, c)
                path.append(u)
            else:
                break

        # expansion of the tree
        if u.moves:
            u = expand(tree, u)
            path.append(u)
            while len(u.moves) == 1:
                u = tree.add_child(u, 0)
                path.append(u)

            if verbose:
                print(f'path {print_path(path)}')

            # add a root connection if the line from u to v is forced
            if forced and len(path) > 3 and useFPN:
                add_root_connection(tree, path)

            # add a connection to a root child if the line from u to v is forced
            if useEnemyFPN and enemyForced and len(path) > 4 and rootChild is not None:
                add_enemy_root_connection(rootChild, path)
                # print("enemyForced, path: " + str(path))

        else:
            if verbose:
                print(f'no expansion possible')

        # simulation
        Delta = simulate(u)

        # backpropagation
        for v in path:
            v.Q += Delta
            v.N += 1

        if verbose:
            log_uct_scores(tree, c)

    return best_child(tree.root(), 0)


def run():
    text = '''
           .   .   .   .   . 
         .   x   x   x   x   
           .   .   .   .   . 
         .   x   .   .   x   
           .   .   .   .   . 
         o   .   .   .   .   
           .   .   .   .   o 
         .   o   o   o   o   
           .   .   .   .   . 
         .   .   .   .   .   W
        '''
    pos = parse_position(text)
    display_position(pos)

    max_iterations = 10000
    tree = MCTSTree(pos)
    c = 1.0 / math.sqrt(2)
    u = mcts_traps(tree, c, max_iterations, False, verbose=True)
    print('=== best move target ===')
    print(print_position(u.state, True, True))


def getRootchildFromNode(u: MCTSNode, tree: MCTSTree):
    while u.parent is not tree.root():
        u = u.parent
    return u


def getMovefromChild(u: MCTSNode, pos: Pos, tree: MCTSTree):
    for move in tree.root().moves:
        if pos.succ(move) == u.state:
            return move
    # for i in range(len(tree.root().children)):
    #     if tree.root().children[i] is u:
    #         return tree.root().moves
    # print("no move found")


def actionFromMove(move, pos: Pos):
    moves = generate_moves(pos)
    for i in range(len(moves)):
        if moves[i] == move:
            action = i
            return action
    else:
        print("problem")
        return -1


def runPos(pos: Pos, iter: int, useFPN = True, useEnemyFPN = False):
    checkPos = copy.deepcopy(pos)
    tree = MCTSTree(pos)
    c = 1.0 / math.sqrt(2)
    u = mcts_traps(tree, c, iter, useFPN, useEnemyFPN, verbose=False)
    rootchild = getRootchildFromNode(u, tree)
    move = getMovefromChild(rootchild, pos, tree)
    action = actionFromMove(move, pos)
    if action == -1:
        for move in tree.root().moves:
            print(print_move(move, checkPos))
            print(pos.succ(move))
        print(u.state)
    # print(action)
    return action


if __name__ == '__main__':
    init_scan()
    run()