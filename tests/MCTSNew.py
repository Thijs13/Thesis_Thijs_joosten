import MCTSNode
import Tree
from draughts1 import *
import time
import random

def MCTS(pos, timeSpan, UCT):
    curWhite = True
    if print_position(pos, False, True)[-1] == "B":
        curWhite = False
    moves = generate_moves(pos)
    root = Node.MCTSNode(curWhite, moves)
    tree = Tree.Tree([root])
    startTime = time.time()
    while startTime + timeSpan > time.time():
        newNode, newPos = tree.searchNewNode(pos, UCT)
        result = randomPlayout(newPos)
        tree.backPropagate(newNode, result)
    bestMove = tree.chooseBestMove(1)



def randomPlayout(pos):
    return 0.5

