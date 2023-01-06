import copy
import math
import random
from typing import Optional

import MCTSNode
from numpy import log as ln
from draughts1 import *
from ete3 import Tree as DisplayTree, TreeStyle, faces, NodeStyle, TextFace

class Tree():

    def __init__(self, nodes: list = None, NHIT: int = 0):
        self.nodes = []
        self.root = None
        self.curNodeName = 0
        self.NHIT = NHIT
        self.backPropJumpNodes = []
        self.nodePath = []
        self.jumpNodeList = []
        self.numFindChildCalls = 0
        self.numJumps = 0
        self.numNormNodes = 0
        self.numJumpNodes = 0
        self.writer = None
        if nodes is not None:
            self.nodes = nodes
            self.root = nodes[0]

    def addNode(self, node: MCTSNode):
        if len(self.nodes) == 0:
            self.root = node
        self.nodes.append(node)

    def createChildNode(self, parent: MCTSNode, move, pos: Pos):
        newPos = pos.succ(move)
        moves = generate_moves(newPos)
        result = self.checkVicDBTree(newPos)
        endPosition = None
        if result is not None:
            endPosition = result
        self.curNodeName += 1
        newNode = MCTSNode.MCTSNode(not parent.getWhite(), moves, None, parent, endPosition, self.curNodeName, print_move(move, pos))
        parent.addChild(newNode, move)
        self.nodes.append(newNode)
        if self.NHIT == 1:
            self.normalNHIT(moves, parent, newNode, newPos)
        elif self.NHIT == 2:
            self.alternateNHIT(newNode, newPos)
        return newNode

    def normalNHIT(self, moves, parent, newNode, newPos):
        if len(moves) > 1 and len(parent.getMoves()) == 1 and not parent == self.root:
            parent.setForcedSeq(1)
            newNode.setBoardState(newPos)
            curNodeChoice = parent.getParent()
            curNodeForced = curNodeChoice.getParent()
            while (not curNodeForced is None) and curNodeForced.getForcedSeq() != 0:
                curNodeChoice.setForcedSeq(curNodeChoice.getForcedSeq() + 2)
                # curNodeChoice.addSeqFinalNode(newNode)
                curNodeForced.setForcedSeq(curNodeForced.getForcedSeq() + 2)
                curNodeForced.addSeqFinalNode(newNode)
                self.jumpNodeList.append(newNode)
                curNodeChoice = curNodeForced.getParent()
                curNodeForced = curNodeChoice.getParent()

    def alternateNHIT(self, newNode, newPos):
        self.numNormNodes += 1
        if newNode.getWhite() != self.root.getWhite() and newNode.getParent() is not self.root:
            node = newNode.getParent()
            curNodeChoice = True
            while not node == self.root:
                if curNodeChoice and len(node.getChildren()) == 1:
                    return
                elif not curNodeChoice and len(node.getChildren()) != 1:
                    return
                else:
                    node = node.getParent()
                    curNodeChoice = not curNodeChoice
            newNode.setBoardState(newPos)
            self.numJumpNodes += 1
            self.jumpNodeList.append(newNode)

    def searchNewNode(self, pos: Pos, UCT: bool):
        curNode = self.root
        newPos = pos
        while True:
            self.numFindChildCalls += 1
            if curNode.getEndPosition() is not None:
                self.finalNodeFound(curNode)
                return curNode, newPos
            unexMove = curNode.getUnexploredMove()
            if unexMove is not None:
                newChild = self.createChildNode(curNode, unexMove, newPos)
                self.finalNodeFound(newChild)
                newPos = newPos.succ(unexMove)
                while len(newChild.getMoves()) == 1:
                    newMove = newChild.getMoves()[0]
                    newChild = self.createChildNode(newChild, newMove, newPos)
                    self.finalNodeFound(newChild)
                    newPos = newPos.succ(newMove)
                return newChild, newPos
            else:
                curNode, curMove, jumpNode = self.selectChildNode(curNode, UCT)
                if not jumpNode:
                    newPos = newPos.succ(curMove)
                else:
                    newPos = curNode.getBoardState()
                    self.numJumps += 1
                self.storePath(curNode, False, jumpNode)

    def storePath(self, node, endPath, jump = False):
        if node is not None:
            self.nodePath.append((node, jump))
        if endPath:
            nodeNames = ""
            nodeMoves = ""
            for list in self.nodePath:
                node = list[0]
                nodeNames = nodeNames + " " + str(node.getName())
                if list[1] == True:
                    rootPathText = str(node.getName())
                    parent = node.getParent()
                    while parent != self.root:
                        rootPathText = str(parent.getName()) + " " + rootPathText
                        parent = parent.getParent()
                    nodeMoves = nodeMoves + "new jump " + str(print_position(node.getBoardState(), False, True)) + " path to root ("\
                                + rootPathText + ")"
                else:
                    nodeMoves = nodeMoves + " " + str(node.getTxtMove())
            self.nodePath = []
            text = ""
            text += "path " + nodeMoves + "\n"
            text += "uct scores \n"
            parSims = self.root.getNumSims()
            for child in self.root.getChildren():
                if self.root.getWhite():
                    curWins = child.getWhiteWins()
                else:
                    curWins = child.getBlackWins()
                curSims = child.getNumSims()
                curUCT = round(curWins / curSims + 1 / math.sqrt(2) * math.sqrt(2 * ln(parSims) / curSims), 4)
                text += str(child.getName()) + " " + child.getTxtMove() + " " + "uct = " + str(curUCT) + " N = " + str(curSims) + "\n"
            for child in self.jumpNodeList:
                if self.root.getWhite():
                    curWins = child.getWhiteWins()
                else:
                    curWins = child.getBlackWins()
                curSims = child.getNumSims()
                curUCT = round(curWins / curSims + 1 / math.sqrt(2) * math.sqrt(2 * ln(parSims) / curSims), 4)
                text += str(child.getName()) + " jump(" + str(print_position(child.getBoardState(), False, True)) + \
                        ") " + "uct = " + str(curUCT) + " N = " + str(curSims) + "\n"
            text += "\n"
            if self.writer is not None:
                self.writer.write(text)

    def finalNodeFound(self, curNode):
        self.storePath(curNode, False)

    def selectChildNode(self, node: MCTSNode, UCT: bool):
        if not UCT:
            childMoveList = node.getChildMoveList()
            ranIndex = random.randint(0, len(childMoveList) - 1)
            return childMoveList[ranIndex][0], childMoveList[ranIndex][1]
        else:
            jumpNode = False
            parSims = node.getNumSims()
            bestUCT = 0
            initialNode = node.getChildren(False)[0]
            bestChild = initialNode
            bestMove = node.getMoveFromChild(initialNode)
            if self.NHIT == 1:
                topNHIT = False
                if not node == self.root:
                    topNHIT = node.getParent().getForcedSeq() == 0
                nodeList = node.getChildren(topNHIT)
            elif self.NHIT == 2:
                nodeList = node.getChildren(False).copy()
                if node == self.root:
                    for jumpNode in self.jumpNodeList:
                        if (not node.getWhite() == jumpNode.getWhite()) and jumpNode.getNumSims() > 0:
                            nodeList.append(jumpNode)
            else:
                nodeList = node.getChildren(False)
            for child in nodeList:
                if node.getWhite():
                    curWins = child.getWhiteWins()
                else:
                    curWins = child.getBlackWins()
                curSims = child.getNumSims()
                curUCT = curWins / curSims + 1 / math.sqrt(2) * math.sqrt(2 * ln(parSims) / curSims)
                if curUCT > bestUCT:
                    bestUCT = curUCT
                    bestChild = child
                    bestMove = node.getMoveFromChild(child)
                    if self.NHIT == 1:
                        if bestChild in node.getSeqFinalNodes():
                            jumpNode = True
                        else:
                            jumpNode = False
                    elif self.NHIT == 2:
                        if node == self.root and (not bestChild in self.root.getChildren()):
                            jumpNode = True
                        else:
                            jumpNode = False
            if jumpNode:
                self.backPropJumpNodes.append([bestChild, node])
            return bestChild, bestMove, jumpNode

    def backPropagate(self, node: MCTSNode, result: int):
        curNode = node
        curNode.updateWins(result)
        jumped = False
        endSkipNode = None
        while not curNode == self.root:
            for nodePair in self.backPropJumpNodes:
                if curNode == nodePair[0]:
                    jumped = True
                    endSkipNode = nodePair[1]
            if jumped:
                while not curNode == endSkipNode:
                    curNode = curNode.getParent()
                jumped = False
            else:
                curNode = curNode.getParent()
            curNode.updateWins(result)
        self.backPropJumpNodes = []

    def chooseBestMove(self, selectionMethod: str = "average"):
        bestChild = None
        bestResult = -math.inf
        rootChildren = self.root.getChildren(False)
        for child in rootChildren:
            if selectionMethod == "average":
                curResult = self.getNodeAverage(child)
            elif selectionMethod == "wins":
                curResult = self.getNodeWins(child)
            elif selectionMethod == "sims":
                curResult = child.getNumSims()
            if curResult > bestResult:
                bestChild = child
                bestResult = curResult
        return self.root.getMoveFromChild(bestChild)

    # Gets the average from the perspective of the parent node
    def getNodeAverage(self, node: MCTSNode) -> float:
        totalSims = node.getNumSims()
        if not node.getWhite():
            return node.getWhiteWins() / totalSims
        else:
            return node.getBlackWins() / totalSims

    # Gets the average from the perspective of the parent node
    def getNodeWins(self, node: MCTSNode) -> float:
        if not node.getWhite():
            return node.getWhiteWins()
        else:
            return node.getBlackWins()

    def checkVictoryTree(self, pos: Pos) -> Optional[float]:
        if not pos.can_move(Side.White):
            return -1
        elif not pos.can_move(Side.Black):
            return 1
        else:
            return None

    def checkVicDBTree(self, pos: Pos) -> Optional[float]:
        numWhite = pos.white_man_count() + pos.white_king_count()
        numBlack = pos.black_man_count() + pos.black_king_count()
        if numWhite > 0 and numBlack > 0 and numWhite + numBlack <= 6:
            curWhite = pos.is_white_to_move()
            value = EGDB.probe(pos)
            if (curWhite and value == 2) or (not curWhite and value == 1):
                return 1
            elif (not curWhite and value == 2) or (curWhite and value == 1):
                return -1
            else:
                return 0
        else:
            return self.checkVictoryTree(pos)

    def display(self):
        displayRoot = DisplayTree()
        displayRoot.add_face(TextFace(str(self.root.getNumSims()), fsize=250), column=0, position='branch-top')
        self.root.setDisplayNode(displayRoot)
        nodeList = [[self.root, False]]
        while len(nodeList) > 0:
            curNode = nodeList[0][0]
            curNodeJump = False
            if nodeList[0][1]:
                curNodeJump = True
            if not curNodeJump:
                curChildren = []
                for child in curNode.getChildren(False):
                    curChildren.append([child, False])
                for jumpChild in curNode.getSeqFinalNodes():
                    if curNode.getParent().getForcedSeq() == 0:
                        # print("grandparent")
                        curChildren.append([jumpChild, True])
                for child in curChildren:
                    nodeList.append(child)
                    # numSims = child[0].getNumSims()
                    # nodeText = "s:" + str(numSims)
                    # nodeText = str(child[0].getName()) + " " + str(child[0].getForcedSeq())
                    nodeText = str(child[0].getName()) + " " + str(child[0].getTxtMove())
                    # if child[0].getParent().getWhite():
                    #     curWins = child[0].getWhiteWins()
                    # else:
                    #     curWins = child[0].getBlackWins()
                    # curSims = child[0].getNumSims()
                    # curUCT = curWins / curSims + math.sqrt(2) * math.sqrt(2 * ln(child[0].getParent().getNumSims()) / curSims)
                    # nodeText = str(curSims) + " " + str(round(curUCT, 3))
                    # nodeText = str(child[0].getForcedSeq())
                    nstyle = NodeStyle()
                    if child[1]:
                        # print("jumpDisplayNode")
                        nstyle["fgcolor"] = "green"
                    else:
                        nstyle["fgcolor"] = "blue"
                    nstyle["size"] = 500
                    face = TextFace(nodeText, fsize=250)
                    face.rotable = True
                    node = curNode.getDisplayNode().add_child(name=nodeText)
                    node.set_style(nstyle)
                    node.add_face(face, column=0, position='branch-top')
                    if not child[1]:
                        child[0].setDisplayNode(node)
            nodeList.pop(0)
        ts = TreeStyle()
        ts.show_leaf_name = True
        ts.rotation = 90
        ts.scale = 5000
        displayRoot.show(tree_style=ts)

