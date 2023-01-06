from draughts1 import *

class DraughtsPuzzle():

    def __init__(self, index, kingDepth, kingPv, kingValue, level, solution, pos, scanDepth = None, scanPv = None, scanValue = None):
        self.index = index
        self.kingDepth = kingDepth
        self.kingPv = kingPv
        self.kingValue = kingValue
        self.level = level
        self.pos = parse_position(pos)
        self.scanDepth = scanDepth
        self.scanPv = scanPv
        self.scanValue = scanValue
        self.solution = []
        movesLeft = True
        while movesLeft:
            solPar = solution.partition(" ")
            if solPar[0] == "":
                movesLeft = False
            else:
                self.solution.append(solPar[0].rstrip())
                solution = solPar[2]

    def getIndex(self):
        return self.index

    def getKingPv(self):
        return self.kingPv

    def getLevel(self):
        return self.level

    def getPos(self):
        return self.pos

    def getValue(self):
        return self.scanValue

    def getSolution(self):
        return self.solution

    def checkSolution(self, moveList):
        if len(moveList) != len(self.solution):
            return False
        else:
            for i in range(len(moveList)):
                if not str(moveList[i]) == str(self.solution[i]):
                    return False
            return True

    def checkFirstMove(self, move):
        if not str(move) == str(self.solution[0]):
            return False
        else:
            return True

