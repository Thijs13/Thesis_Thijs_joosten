import math
from typing import Optional

from draughts1 import *
import random
import time
import func_timeout
import numpy as np
import cv2 as cv
from waiting import wait
from ete3 import Tree, TreeStyle, faces, NodeStyle, TextFace
import Tree
import MCTSNode
import HashTable
import NeuralNetworks
import csv
from keras.models import Sequential
# from keras.layers import Dense, convolutional
import keras.layers as layers
import tensorflow as tf

class Players():

    def __init__(self):
        self.addRandom = False
        self.xClick = 0
        self.yClick = 0
        # Variables for diagnostics
        self.MCTSCalls = 0
        self.numNodes = 0
        self.treeTraversalTime = 0
        self.playoutTime = 0
        self.backpropTime = 0
        self.playoutGetMovesTime = 0
        self.playoutPlayMovesTime = 0
        self.playoutCheckTime = 0
        self.playoutNumGetMoves = 0
        self.checkVicEvalTime = 0
        self.printPosTime = 0
        self.tree = None

    def human(self, pos, mouse):
        print("Evaluation: " + str(eval_position(pos)))
        moves = generate_moves(pos)
        count = 0
        if mouse:
            wait(lambda: self.mouseVarsChanged(), timeout_seconds=1000)
            print("human mouse input")
        else:
            for move in moves:
                print(str(count) + ": " + print_move(move, pos))
                count += 1
            inputCor = False
            while not inputCor:
                curInput = input("Enter move number:")
                if curInput.isnumeric():
                    curInput = int(curInput)
                    if curInput >= 0 and curInput < len(moves):
                        inputCor = True
                        action = curInput
                    else:
                        print("Not a valid move")
                else:
                    print("Not a valid input")
            return action

    # Function that is called in case of a mouse event
    def humanMouse(self, event, x, y, flags, param):
        print("mouse")
        if event == cv.EVENT_LBUTTONDOWN:
            self.xClick = x
            self.yClick = y
            print("Single click, x: " + str(x) + ", y: " + str(y))

    def mouseVarsChanged(self):
        if not self.xClick == 0:
            return True
        return False

    def timedMiniMax(self, pos, timeSpan, minimaxAlg=0):
        startTime = time.time()
        notDone = True
        depth = 1
        bestEval, bestMove = None, None
        if pos.is_white_to_move():
            maxPlayer = True
        else:
            maxPlayer = False
        while notDone:
            timeLeft = (startTime + timeSpan) - time.time()
            if timeLeft < 0:
                notDone = False
            else:
                try:
                    if minimaxAlg == 0:
                        bestEval, bestMove = func_timeout.func_timeout(timeLeft, self.alphaBeta,
                                                                       args=[pos, depth, maxPlayer, -math.inf, math.inf])
                    elif minimaxAlg == 1:
                        bestEval, bestMove = func_timeout.func_timeout(timeLeft, self.miniMax, args=[pos, depth, maxPlayer])
                    else:
                        bestEval, bestMove = func_timeout.func_timeout(timeLeft, self.shuflleMinimax, args=[pos, depth])
                except:
                    notDone = False
            depth += 1
        action = self.actionFromMove(bestMove, pos)
        return action

    def miniMax(self, pos, depth, maxPlayer):
        moves = generate_moves(pos)
        bestEval = None
        bestMove = None
        if depth == 0 or len(moves) == 0:
            checkWin = self.checkVicDB(pos)
            if checkWin == None:
                if print_position(pos, False, True)[-1] == "W":
                    return eval_position(pos), None
                else:
                    return eval_position(pos) * -1, None
            else:
                return checkWin * 1000, None
        if len(moves) == 1:
            curEval, discardMove = self.miniMax(pos.succ(moves[0]), depth, not maxPlayer)
            return curEval, moves[0]
        if maxPlayer:
            evalList = []
            for move in moves:
                curEval, discardMove = self.miniMax(pos.succ(move), depth - 1, not maxPlayer)
                evalList.append(curEval)
            for i in range(len(evalList)):
                if bestEval is None or bestEval < evalList[i]:
                    bestEval = evalList[i]
                    bestMove = moves[i]
        else:
            evalList = []
            for move in moves:
                curEval, discardMove = self.miniMax(pos.succ(move), depth - 1, not maxPlayer)
                evalList.append(curEval)
            for i in range(len(evalList)):
                if bestEval is None or bestEval > evalList[i]:
                    bestEval = evalList[i]
                    bestMove = moves[i]
        return bestEval, bestMove

    def useAlphaBeta(self, pos: Pos, depth: int, maxPlayer: bool, alpha: float, beta: float):
        bestEval, bestMove = self.alphaBeta(pos, depth, maxPlayer, alpha, beta)
        action =  self.actionFromMove(bestMove, pos)
        return action

    def alphaBeta(self, pos, depth, maxPlayer, alpha, beta):
        moves = generate_moves(pos)
        bestEval = None
        bestMove = None
        if depth == 0 or len(moves) == 0:
            checkWin = self.checkVicDB(pos)
            if checkWin == None:
                if print_position(pos, False, True)[-1] == "W":
                    return eval_position(pos), None
                else:
                    return eval_position(pos) * -1, None
            else:
                return checkWin * 1000, None
        if len(moves) == 1:
            curEval, discardMove = self.alphaBeta(pos.succ(moves[0]), depth, not maxPlayer, alpha, beta)
            return curEval, moves[0]
        if maxPlayer:
            for move in moves:
                curEval, discardMove = self.alphaBeta(pos.succ(move), depth - 1, not maxPlayer, alpha, beta)
                if bestEval is None or bestEval < curEval:
                    bestEval = curEval
                    bestMove = move
                if self.addRandom and bestEval == curEval:
                    if random.randint(0, 2) == 0:
                        bestEval = curEval
                        bestMove = move
                if bestEval >= beta:
                    break
                if alpha < bestEval:
                    alpha = bestEval
        else:
            for move in moves:
                curEval, discardMove = self.alphaBeta(pos.succ(move), depth - 1, not maxPlayer, alpha, beta)
                if bestEval is None or bestEval > curEval:
                    bestEval = curEval
                    bestMove = move
                if self.addRandom and bestEval == curEval:
                    if random.randint(0, 2) == 0:
                        bestEval = curEval
                        bestMove = move
                if bestEval <= alpha:
                    break
                if beta > bestEval:
                    beta = bestEval
        return bestEval, bestMove

    def simpleMinimax(self, pos: Pos, depth: int):
        score, move = minimax_search(pos, depth)
        action = self.actionFromMove(move, pos)
        return action

    def scanMinimax(self, pos: Pos, depth: int, time: float):
        score, move = scan_search(pos, depth, time)
        action = self.actionFromMove(move, pos)
        return action

    def shuflleMinimax(self, pos: Pos, depth: int):
        score, move = minimax_search_with_shuffle(pos, depth)
        action = self.actionFromMove(move, pos)
        return action

    # Runs the MCTS algorithm. Takes as input the current board state, maximum execution time, evaluation algorithm,
    # maximum depth of evaluation algorithm (if relevant), use UCT or not, evaluation methode of final move, use hastable or not
    # maximum number of nodes to use (None for infinite), neuralnetwork to evaluate, use minimax in selection or not,
    # use NHIT (now FPN) or not, a writer to record.
    # Returns the best move
    def MCTS(self, pos: Pos, timeSpan: float, playout: str, depth: int, UCT: bool, evalMode: str='average', hashTable: HashTable=None,
             maxNodes: int=None, neuralNetworks: NeuralNetworks=None, selection: bool = False, NHIT:int = 0, writer = None):
        self.MCTSCalls += 1
        curWhite = pos.is_white_to_move()
        moves = generate_moves(pos)
        root = MCTSNode.MCTSNode(curWhite, moves)
        self.tree = Tree.Tree([root], NHIT)
        if writer is not None:
            self.tree.writer = writer
        if neuralNetworks is not None:
            corValue, corMove = self.alphaBeta(pos, 5, pos.is_white_to_move(), -math.inf, math.inf)
            neuralNetworks.testValueNetwork(pos, corValue, corMove, True)
            neuralNetworks.testNeuralNetwork(pos)
        startTime = time.time()
        nodeCount = 0
        while startTime + timeSpan > time.time():
            self.numNodes += 1
            newNode, newPos = self.tree.searchNewNode(pos, UCT)
            result = None
            if selection:
                result = self.minimaxInSelection(newPos, 3)
            if result is None:
                if playout == "random":
                    result = playout_random(newPos, 150, False)
                elif playout == "naive":
                    result = self.naivePlayout(newPos)
                elif playout == "minimaxPlayout":
                    result = playout_minimax(newPos, depth, 1, 150, 1000000000000, False)
                elif playout == "simpleMinimax":
                    result = self.simpleMinimaxPlayoutReplace(newPos, depth)
                elif playout == "scanMinimax":
                    result = self.simpleMinimaxPlayoutReplace(newPos, depth)
                elif playout == "scanContinuous":
                    result = self.scanSimContinuous(newPos)
                elif playout == "scanDiscreet":
                    result = self.scanSimDiscreet(newPos)
                else:
                    print("No playout selected")
            self.tree.backPropagate(newNode, result)
            self.tree.storePath(None, True)
            nodeCount += 1
            if maxNodes is not None and nodeCount >= maxNodes:
                bestMove = self.tree.chooseBestMove(evalMode)
                action = self.actionFromMove(bestMove, pos)
                return action
        bestMove = self.tree.chooseBestMove(evalMode)
        action = self.actionFromMove(bestMove, pos)
        return action

    def randomPlayout(self, pos: Pos) -> Optional[float]:
        playPos = pos
        moveCount = 0
        while True:
            # playoutCheckTimeStart = time.time()
            checkVic = self.checkVicDB(playPos)
            # self.playoutCheckTime += time.time() - playoutCheckTimeStart
            if not checkVic == None:
                return checkVic
            elif moveCount > 100:
                return 0
            else:
                moveCount = moveCount + 1
                # self.playoutNumGetMoves += 1
                # playoutGetMovesTimeStart = time.time()
                moves = generate_moves(playPos)
                # self.playoutGetMovesTime += time.time() - playoutGetMovesTimeStart
                n = random.randint(0, len(moves) - 1)
                # playoutplayMovesTimeStart = time.time()
                playPos = playPos.succ(moves[n])
                # self.playoutPlayMovesTime += time.time() - playoutplayMovesTimeStart

    def minimaxPlayout(self, pos: Pos, depth: int) -> Optional[float]:
        playPos = pos
        moveCount = 0
        while True:
            playoutCheckTimeStart = time.time()
            checkVic = self.checkVicDB(playPos)
            self.playoutCheckTime += time.time() - playoutCheckTimeStart
            if not checkVic == None:
                return checkVic
            elif moveCount > 100:
                return 0
            else:
                moveCount = moveCount + 1
                self.playoutNumGetMoves += 1
                if print_position(pos, False, True)[-1] == "W":
                    maxPlayer = True
                else:
                    maxPlayer = False
                playoutGetMovesTimeStart = time.time()
                bestEval, bestMove = self.alphaBeta(playPos, depth, maxPlayer, -math.inf, math.inf)
                self.playoutGetMovesTime += time.time() - playoutGetMovesTimeStart
                playoutplayMovesTimeStart = time.time()
                playPos = playPos.succ(bestMove)
                self.playoutPlayMovesTime += time.time() - playoutplayMovesTimeStart

    def naivePlayout(self, pos: Pos) -> Optional[float]:
        value = piece_count_eval(play_forced_moves(pos))
        if not pos.is_white_to_move():
            value = value * -1
        if value > 0:
            return 1
        elif value < 0:
            return -1
        else:
            return 0

    def simpleMinimaxPlayoutReplace(self, pos: Pos, depth: int) -> Optional[float]:
        value, move = minimax_search_with_shuffle(pos, depth)
        if not pos.is_white_to_move():
            value = value * -1
        if value > 0:
            return 1
        elif value < 0:
            return -1
        else:
            return 0

    def scanMinimaxPlayoutReplace(self, pos: Pos, depth: int) -> Optional[float]:
        value, move = scan_search(pos, depth, 10)
        if not pos.is_white_to_move():
            value = value * -1
        if value > 50:
            return 1
        elif value < -50:
            return -1
        else:
            return 0

    def scanSimDiscreet(self, pos: Pos):
        value = eval_position(pos)
        if value > 50:
            return 1
        elif value < -50:
            return -1
        else:
            return 0

    def scanSimContinuous(self, pos: Pos) -> Optional[float]:
        value = eval_position(pos)
        reward = 1 / (1 + math.exp(-(value/100)))
        return reward

    def minimaxInSelection(self, pos: Pos, depth: int) -> Optional[float]:
        curEval = eval_position(pos)
        # curWhite = pos.is_white_to_move()
        # bestEval, bestMove = self.alphaBeta(pos, depth, curWhite, -math.inf, math.inf)
        # print("curWhite: " + str(curWhite))
        bestEval, move = scan_search(pos, depth, 5)
        # print("curEval: " + str(curEval))
        # print("bestEval: " + str(bestEval))
        if bestEval - curEval >= 300:
            return 1
        elif bestEval - curEval <= -300:
            return -1
        # elif not curWhite and bestEval - curEval <= -100:
        #     return 1
        # elif not curWhite and bestEval - curEval >= 100:
        #     return -1

    def getForcedSeqInfo(self, tree: Tree, depth: int, pos: Pos):
        root = tree.root
        curWhite = root.getWhite()
        for index, child in enumerate(root.getChildren()):
            forcedSeqsPlayer = self.searchForcedSeq(child, depth - 1, not curWhite)
            forcedSeqsEnemy = self.searchForcedSeq(child, depth - 1, curWhite)
            if forcedSeqsPlayer is None:
                forcedSeqsPlayer = []
            if forcedSeqsEnemy is None:
                forcedSeqsEnemy = []
            print("Move " + str(index) + ": " + str(print_move(root.getMoveFromChild(child), pos)))
            print("There are " + str(len(forcedSeqsPlayer)) + " forced sequences")
            print("There are " + str(len(forcedSeqsEnemy)) + " forced sequences for the enemy")
            print("The are " + str(child.getNumSims()) + " simulations for this node")

    def searchForcedSeq(self, node: MCTSNode, depth: int, whiteForced: bool):
        children = node.getChildren()
        if depth == 0:
            return [[node]]
        if node.endPosition is not None:
            return None
        if whiteForced and node.white and len(children) > 1:
            return None
        if not whiteForced and not node.white and len(children) > 1:
            return None
        forcedSeqsList = []
        for child in children:
            forcedSeqs = self.searchForcedSeq(child, depth - 1, whiteForced)
            if forcedSeqs is not None:
                for forcedSeq in forcedSeqs:
                    forcedSeq.insert(0, node)
                    forcedSeqsList.append(forcedSeq)
        if len(forcedSeqsList) == 0:
            return None
        return forcedSeqsList

    def checkVictory(self, pos: Pos) -> Optional[float]:
        if not pos.can_move(Side.White):
            return -1
        elif not pos.can_move(Side.Black):
            return 1
        else:
            return None

    def checkVicDB(self, pos) -> Optional[float]:
        printPosTimeStart = time.time()
        self.printPosTime += time.time() - printPosTimeStart
        numWhite = pos.white_man_count() + pos.white_king_count()
        numBlack = pos.black_man_count() + pos.black_king_count()
        if numWhite > 0 and numBlack > 0 and numWhite + numBlack <= 6:
            curWhite = pos.is_white_to_move()
            checkVicEvalTimeStart = time.time()
            value = EGDB.probe(pos)
            self.checkVicEvalTime += time.time() - checkVicEvalTimeStart
            if (curWhite and value == 2) or (not curWhite and value == 1):
                return 1
            elif (not curWhite and value == 2) or (curWhite and value == 1):
                return -1
            else:
                return 0
        else:
            return self.checkVictory(pos)

    def actionFromMove(self, move, pos: Pos) -> int:
        moves = generate_moves(pos)
        for i in range(len(moves)):
            if moves[i] == move:
                action = i
                # print("Zet " + str(i))
                return action
        else:
            print("problem")
            return -1

    def getScanSearchOutput(self, pos: Pos):
        si = SearchInput()
        si.move = True
        si.book = False
        si.depth = 14
        si.nodes = 1000000000000
        si.time = 5.0
        si.input = True
        si.output = OutputType.Terminal

        so = SearchOutput()
        node = make_node(pos)
        search(so, node, si)
        # print(so.score)
        print(si.moves)
        return so.move

    def printTime(self):
        print("")
        print("MCTS tree traversal time: " + str(self.treeTraversalTime))
        print("MCTS playout time: " + str(self.playoutTime))
        print("MCTS backpropagation time: " + str(self.backpropTime))
        print("MCTS get moves playout time: " + str(self.playoutGetMovesTime))
        print("MCTS play moves playout time: " + str(self.playoutPlayMovesTime))
        print("MCTS check playout time: " + str(self.playoutCheckTime))
        print("MCTS check scan evaluation time: " + str(self.checkVicEvalTime))
        print("MCTS position to text time: " + str(self.printPosTime))
        print("")
        print("MCTS tree traversal per MCTS call time: " + str(self.treeTraversalTime / self.MCTSCalls))
        print("MCTS playout per MCTS call time: " + str(self.playoutTime / self.MCTSCalls))
        print("MCTS backpropagation per MCTS call time: " + str(self.backpropTime / self.MCTSCalls))
        print("MCTS get moves playout per MCTS call time: " + str(self.playoutGetMovesTime / self.MCTSCalls))
        print("MCTS play moves playout per MCTS call time: " + str(self.playoutPlayMovesTime / self.MCTSCalls))
        print("MCTS check playout per MCTS call time: " + str(self.playoutCheckTime / self.MCTSCalls))
        print("MCTS check scan evaluation per MCTS call time: " + str(self.checkVicEvalTime / self.MCTSCalls))
        print("MCTS position to text per MCTS call time: " + str(self.printPosTime / self.MCTSCalls))
        print("")
        print("MCTS tree traversal per node time: " + str(self.treeTraversalTime / self.numNodes))
        print("MCTS playout per node time: " + str(self.playoutTime / self.numNodes))
        print("MCTS backpropagation per node time: " + str(self.backpropTime / self.numNodes))
        print("MCTS get moves playout per node time: " + str(self.playoutGetMovesTime / self.numNodes))
        print("MCTS play moves playout per node time: " + str(self.playoutPlayMovesTime / self.numNodes))
        print("MCTS check playout per node time: " + str(self.playoutCheckTime / self.numNodes))
        print("MCTS check scan evaluation per node time: " + str(self.checkVicEvalTime / self.numNodes))
        print("MCTS position to text per node time: " + str(self.printPosTime / self.numNodes))
        print("")
        print("MCTS get moves playout per move time: " + str(self.playoutGetMovesTime / self.playoutNumGetMoves))
        print("MCTS play moves playout per move time: " + str(self.playoutPlayMovesTime / self.playoutNumGetMoves))
        print("MCTS check playout per move time: " + str(self.playoutCheckTime / self.playoutNumGetMoves))
        print("MCTS check scan evaluation per move time: " + str(self.checkVicEvalTime / self.playoutNumGetMoves))
        print("MCTS position to text per move time: " + str(self.printPosTime / self.playoutNumGetMoves))
