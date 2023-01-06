import random
from draughts1 import *

class HashTable:
    boardValueTable = {}
    hashTable = {}

    def __init__(self, boardValueTable = None):
        if boardValueTable is None:
            self.fillBoardValueTable()

    def fillBoardValueTable(self):
        pieces = ["x", "o", "X", "O"]
        for piece in pieces:
            for i in range(50):
                randomVal = random.randint(1000000000, 9999999999)
                self.boardValueTable[(piece, i)] = randomVal

    def computeHashValue(self, pos):
        posText = print_position(pos, False, True)[:-1]
        hashValue = 0
        for i in range(len(posText)):
            if posText[i] != ".":
                hashValue = hashValue ^ self.boardValueTable[(posText[i], i)]
        return hashValue

    def addToHashTable(self, hashValue, value):
        self.hashTable[hashValue] = value

    def checkInHashTable(self, hashValue):
        for hv in self.hashTable:
            if hv == hashValue:
                return self.hashTable[hv]
        return None
