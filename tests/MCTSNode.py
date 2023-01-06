from draughts1 import *
import random

class MCTSNode():

    # name = 0
    # children = []
    # parent = None
    # moves = []
    # childMoveList = []
    # whiteWins = 0.0
    # blackWins = 0.0
    # white = None
    # endPosition = None

    def __init__(self, whiteMove: bool, moves=None, children: list=None, parent=None, endPosition: bool = None, name: int = 0, txtMove: str = ""):
        self.children = []
        self.parent = None
        self.moves = []
        self.childMoveList = []
        self.whiteWins = 0.0
        self.blackWins = 0.0
        self.white = None
        self.endPosition = None
        self.white = whiteMove
        if children is not None:
            self.children = children
        if parent is not None:
            self.parent = parent
        if moves is not None:
            for move in moves:
                self.moves.append(move)
        if children is not None and moves is not None:
            for count, child in enumerate(children):
                self.childMoveList.append([child, moves[count]])
        self.endPosition = endPosition
        self.displayNode = None
        self.forcedSeq = 0
        self.seqFinalNodes = []
        self.boardState = None
        self.name = name
        self.txtMove = txtMove

    def getChildren(self, incTrapSeq: bool=False):
        if not incTrapSeq:
            return self.children
        else:
            jumpNodes = []
            jumpNodes.extend(self.getSeqFinalNodes())
            jumpNodes.extend(self.children)
            return jumpNodes

    def getParent(self):
        return self.parent

    def getMoves(self) -> list:
        return self.moves

    def getChildMoveList(self) -> list:
        return self.childMoveList

    def getWhite(self) -> bool:
        return self.white

    def getNumSims(self) -> float:
        return self.whiteWins + self.blackWins

    def getWhiteWins(self) -> float:
        return self.whiteWins

    def getBlackWins(self) -> float:
        return self.blackWins

    def getEndPosition(self) -> bool:
        return self.endPosition

    def setEndPosition(self, endPosition: bool):
        self.endPosition = endPosition

    def getDisplayNode(self):
        return self.displayNode

    def setDisplayNode(self, displayNode):
        self.displayNode = displayNode

    def getForcedSeq(self):
        return self.forcedSeq

    def setForcedSeq(self, forcedSeq):
        self.forcedSeq = forcedSeq

    def getSeqFinalNodes(self):
        return self.seqFinalNodes

    def setSeqFinalNode(self, seqFinalNodes):
        self.seqFinalNodes = seqFinalNodes

    def addSeqFinalNode(self, seqFinalNode):
        self.seqFinalNodes.append(seqFinalNode)

    def getBoardState(self):
        return self.boardState

    def setBoardState(self, boardState):
        self.boardState = boardState

    def getName(self):
        return self.name

    def getTxtMove(self):
        return self.txtMove

    def addChild(self, child, move):
        self.children.append(child)
        if move not in self.moves:
            self.moves.append(move)
        self.childMoveList.append([child, move])

    def getMoveFromChild(self, child):
        for cmPair in self.childMoveList:
            if cmPair[0] == child:
                return cmPair[1]

    def updateWins(self, result: int):
        if result == 1:
            self.whiteWins += 1
        elif result == -1:
            self.blackWins += 1
        else:
            self.whiteWins += 0.5
            self.blackWins += 0.5

    def getUnexploredMove(self):
        possibleMoves = []
        if len(self.moves) > len(self.childMoveList):
            for move in self.moves:
                for pair in self.childMoveList:
                    if move == pair[1]:
                        break
                else:
                    possibleMoves.append(move)
        if len(possibleMoves) > 0:
            ranIndex = random.randint(0, len(possibleMoves)-1)
            return possibleMoves[ranIndex]
