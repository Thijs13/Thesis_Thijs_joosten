import math

import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from gym import Env, spaces
import time
from draughts1 import *
import Players
import DraughtsPuzzle
import HashTable
import NeuralNetworks
from tests import MCTSWieger
from typing import List
import ToPDN

font = cv2.FONT_HERSHEY_COMPLEX_SMALL


class DraughtsScape(Env):
    def __init__(self):
        super(DraughtsScape, self).__init__()

        # Define a 2-D observation space
        self.observation_shape = (600, 800, 3)
        self.observation_space = spaces.Box(low=np.zeros(self.observation_shape),
                                            high=np.ones(self.observation_shape),
                                            dtype=np.float16)

        # Define an action space ranging from 0 to 4
        # self.action_space = spaces.Discrete(6, )

        # Create a canvas to render the environment images upon
        self.canvas = np.ones(self.observation_shape) * 1

        self.text = ""
        self.pos = Pos()
        self.moveList = []

    @property
    def action_space(self):
        # return generate_moves(self.pos)
        # pos = parse_position(self.text)
        numActions = len(generate_moves(self.pos))
        return spaces.Discrete(numActions)

    def draw_elements_on_canvas(self):
        # Init the canvas
        xOffset = 150
        yOffset = 50

        self.canvas = np.ones(self.observation_shape) * 1

        icon = cv2.imread("images/dambord.jpg") / 255
        board_w = 500
        board_h = 500
        icon = cv2.resize(icon, (board_h, board_w))
        x, y = xOffset, yOffset
        shape = icon.shape
        self.canvas[y: y + shape[1], x:x + shape[0]] = icon

        piece_w = 50
        piece_h = 50
        pieceWhite = cv2.imread("images/checkers_man_white.png") / 255
        pieceBlack = cv2.imread("images/checkers_man_black.png") / 255
        kingWhite = cv2.imread("images/checkers_king_white.png") / 255
        kingBlack = cv2.imread("images/checkers_king_black.png") / 255
        pieceWhite = cv2.resize(pieceWhite, (piece_h, piece_w))
        pieceBlack = cv2.resize(pieceBlack, (piece_h, piece_w))
        kingWhite = cv2.resize(kingWhite, (piece_h, piece_w))
        kingBlack = cv2.resize(kingBlack, (piece_h, piece_w))
        shapePiece = pieceWhite.shape
        posText = print_position(self.pos, False, True)
        for i in range(10):
            for j in range(5):
                x = int(xOffset + (j * 2 + ((i + 1) % 2)) * (board_w / 10))
                y = int(yOffset + i * (board_h / 10))
                curField = posText[i * 5 + j]
                if curField == "x":
                    self.canvas[y: y + shapePiece[1], x:x + shapePiece[0]] = pieceBlack
                elif curField == "o":
                    self.canvas[y: y + shapePiece[1], x:x + shapePiece[0]] = pieceWhite
                elif curField == "X":
                    self.canvas[y: y + shapePiece[1], x:x + shapePiece[0]] = kingBlack
                elif curField == "O":
                    self.canvas[y: y + shapePiece[1], x:x + shapePiece[0]] = kingWhite
                self.canvas = cv2.putText(self.canvas, str(i*5 + j + 1), (x, y + 12), font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

    def reset(self):
        # Reset the reward
        self.ep_return = 0

        # Reset the Canvas
        self.canvas = np.ones(self.observation_shape) * 1

        self.text = '''
                           .   .   .   .   . 
                         .   .   .   x   .   
                           .   x   .   .   . 
                         .   .   .   .   .   
                           .   .   .   o   . 
                         .   o   .   .   .   
                           .   .   .   .   . 
                         .   .   .   .   .   
                           .   .   .   .   . 
                         .   .   .   .   .   W;
                        '''
        self.pos = start_position()

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        # return the observation
        return self.canvas

    def render(self, mode="human", mouseFunction = None):
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("Game", self.canvas)
            if mouseFunction is not None:
                cv2.setMouseCallback("Game", mouseFunction)
            cv2.waitKey(10)

        elif mode == "rgb_array":
            return self.canvas

    def close(self):
        counter = 1
        for txtMove in self.moveList:
            print("Move " + str(counter) + ": " + txtMove)
            counter += 1
        cv2.destroyAllWindows()

    def step(self, action):
        # Flag that marks the termination of an episode
        done = False

        # Assert that it is a valid action
        assert self.action_space.contains(action), "Invalid Action"

        # apply the action
        moves = generate_moves(self.pos)
        self.pos = self.pos.succ(moves[action])

        # Increment the episodic return
        self.ep_return += 1

        # Draw elements on the canvas
        self.draw_elements_on_canvas()

        # If a win/loss/draw, end the episode.
        if not self.pos.can_move(Side.White):
            # print("Win for black")
            reward = -1
            done = True
        elif not self.pos.can_move(Side.Black):
            # print("Win for white")
            reward = 1
            done = True
        else:
            numWhite = self.pos.white_man_count() + self.pos.white_king_count()
            numBlack = self.pos.black_man_count() + self.pos.black_king_count()
            # Why does the number of pieces have to be smaller than 5 instead of 6?
            if numWhite > 0 and numBlack > 0 and numWhite + numBlack <= 6:
                curWhite = self.pos.is_white_to_move()
                value = EGDB.probe(self.pos)
                if (curWhite and value == 2) or (not curWhite and value == 1):
                    reward = 1
                    done = True
                elif (not curWhite and value == 2) or (curWhite and value == 1):
                    reward = -1
                    done = True
                else:
                    reward = 0.5
                    done = True
            else:
                reward = 0
                done = False

        return self.canvas, reward, done, []


def actionFromMove(move, pos):
    moves = generate_moves(pos)
    for i in range(len(moves)):
        if moves[i] == move:
            action = i
            return action
    else:
        print("problem")
        return -1


# Plays maxGames games between the two given players (with lambda) and calculates the elo difference
def playGame(player1, player2, maxGames, players = None):
    env.render()
    startPlayer = True
    curPlayer = startPlayer
    numMove = 0
    numGame = 1
    whiteWins = 0
    blackWins = 0
    prevEval = 0
    trapSetup = 0
    trapNum = 0
    trapStart = 0

    while True:
        numMove += 1
        if curPlayer:
            action = player1()
        else:
            action = player2()

        obs, reward, done, info = env.step(action)

        curPlayer = not curPlayer

        # Render the game
        env.render("human", None)

        if numMove > 400:
            reward = 0.5
            done = True
            print("Draw")

        curEval = piece_count_eval(env.pos)
        if curEval > prevEval and not trapSetup:
            trapSetup = True
            trapStart = prevEval
        elif prevEval == curEval:
            trapSetup = False
        elif curEval < prevEval and trapSetup and curEval < trapStart:
            trapNum += 1
            trapSetup = False
        prevEval = curEval

        # Keep track of wins and calculate ELO
        if done:
            curPlayer = startPlayer
            numMove = 0
            print("result: " + str(reward))
            if reward == 1:
                whiteWins += 1
            elif reward == -1:
                blackWins += 1
            elif reward == 0.5:
                whiteWins += 0.5
                blackWins += 0.5
            expOutcome = whiteWins / numGame
            if 0 < expOutcome < 1:
                eloDif = 400 * math.log((expOutcome / (1 - expOutcome)), 10)
                print("Current ELO difference: " + str(eloDif))
            trapNum = 0
            prevEval = 0
            obs = env.reset()
            if numGame < maxGames:
                numGame += 1
            else:
                break

    expOutcome = whiteWins / numGame
    if 0 < expOutcome < 1:
        eloDif = 400 * math.log((expOutcome / (1 - expOutcome)), 10)
        print("Current ELO difference: " + str(eloDif))
    print("Wins for white: " + str(whiteWins) + ", wins for black: " + str(blackWins))

    env.close()
    return whiteWins > blackWins

# turns the first x puzzles from file into a list of DraughtsPuzzle instances
def readDraughtsPuzzles(startPuzzle, endPuzzle):
    draughtsPuzzleList = []
    # f = open("Wiersma - Damminiaturen.dpf", "r")
    f = open("Letsjinski - Strategia y Taktika.dpf", "r")
    for i in range(startPuzzle, endPuzzle):
        try:
            index = int(f.readline().partition(" = ")[2])
            kingDepth = int(f.readline().partition(" = ")[2])
            kingPv = f.readline().partition(" = ")[2]
            kingValue = float(f.readline().partition(" = ")[2])
            level = int(f.readline().partition(" = ")[2])
            solution = f.readline().partition(" = ")[2]
            position = f.readline().partition(" = ")[2]
            draughtsPuzzle = DraughtsPuzzle.DraughtsPuzzle(index, kingDepth, kingPv, kingValue, level, solution, position)
            draughtsPuzzleList.append(draughtsPuzzle)
            f.readline()
        except Exception as e:
            print(e)
            while True:
                if f.readline() == "\n":
                    break
    f.close()
    return draughtsPuzzleList

# Takes a single puzzle and a player (with lambda) and returns true with the puzzle is correctly solved
# Uses timed minimax as a default opponent
def playDraughtsPuzzle(draughtsPuzzle, show, testPlayer, checkPlayer = None):
    curPlayer = True
    env.pos = draughtsPuzzle.getPos()
    answerMoves = []
    checkFirstMove = True
    firstMoveResult = None
    startEval = eval_position(env.pos)
    for i in range(len(draughtsPuzzle.getKingPv())):
        if curPlayer:
            action = testPlayer()
        else:
            if checkPlayer is not None:
                action = checkPlayer()
            else:
                print("def player")
                action = players.timedMiniMax(env.pos, 0.1, True)

        moves = generate_moves(env.pos)
        answerMoves.append(print_move(moves[action], env.pos))
        if checkFirstMove:
            firstMoveResult = draughtsPuzzle.checkFirstMove(answerMoves[0])

        obs, reward, done, info = env.step(action)

        curPlayer = not curPlayer

        # Render the game
        if show:
            env.render()

        if done:
            if reward == 1:
                return True, firstMoveResult, True
            else:
                return False, firstMoveResult, False

        if draughtsPuzzle.checkSolution(answerMoves):
            endEvalDif = (startEval - eval_position(env.pos)) > 50
            return True, firstMoveResult, endEvalDif
    endEvalDif = (startEval - eval_position(env.pos)) > 50
    return False, firstMoveResult, endEvalDif

# Takes a list of puzzles and an AI (with lambda) and prints the # of correct puzzles
# Uses timed minimax as a default opponent
def checkPuzzles(startPuzzle, endPuzzle, show, testPlayer, checkPlayer = None):
    puzList = readDraughtsPuzzles(startPuzzle, endPuzzle)
    puzCor = 0
    puzWrong = 0
    firstCor = 0
    firstWrong = 0
    difAnswer = 0
    wrongList = []
    puzEvalUp = 0
    for i in range(len(puzList)):
        result, firstMoveResult, endEvalDif = playDraughtsPuzzle(puzList[i], show, testPlayer, checkPlayer)
        if result:
            puzCor += 1
        else:
            puzWrong += 1
            wrongList.append(i + startPuzzle)
        if firstMoveResult:
            firstCor += 1
        else:
            firstWrong += 1
        if endEvalDif:
            puzEvalUp += 1
        if result is not firstMoveResult:
            difAnswer += 1
    print("Puzzles solved correctly: " + str(puzCor))
    print("Puzzles not solved: " + str(puzWrong))
    print("Correct first move: " + str(firstCor))
    print("Incorrect first move: " + str(firstWrong))
    print("Puzzles solved by improved eval: " + str(puzEvalUp))
    print("Different answers: " + str(difAnswer))
    wrongListStr = ""
    for i in range(len(wrongList)):
        wrongListStr = wrongListStr + str(wrongList[i] + 1) + ", "
    print("Index numbers of incorrectly solved puzzles: " + wrongListStr)

def checkWiegerPuzzle():
    textPos1 = "........x.......x.x........o...o....o.............W"
    solution1 = "28-22"
    textPos2 = ".......xx.........x......xoo...o....o.............W"
    solution2 = "27-21"
    textPos3 = ".......xx..........x.....xoo.x........o...o....o..W"
    solution3 = "27-21"
    textPos4 = "........xx.......x.x.....x........o.oooo..........W"
    solution4 = "37-31"
    textPos5 = "......xxxx......x..x.....o........o.oooo..........W"
    solution5 = "26-21"
    textPosList = [textPos1, textPos2, textPos3, textPos4, textPos5]
    solutionList = [solution1, solution2, solution3, solution4, solution5]
    for i in range(0, 5):
        pos = parse_position(textPosList[i])
        solutionFound = 0
        maxNodes = 10
        while solutionFound < 5:
            move = players.MCTS(env.pos, 300, "random", 0, True, 'average', None, maxNodes, None, False, 0)
            if print_move(move, pos) == solutionList[i]:
                solutionFound += 1
            else:
                if maxNodes < 20:
                    maxNodes += 1
                elif maxNodes < 100:
                    maxNodes += 5
                elif maxNodes < 1000:
                    maxNodes += 20
                else:
                    maxNodes += 100
                print(maxNodes)
                solutionFound = 0
        print("Puzzle " + str(i + 1) + " , nodes needed: " + str(maxNodes))

def playPuzzleFive():
    textPos = "......xxxx......x..x.....o........o.oooo..........W"
    pos = parse_position(textPos)
    open("moveLogFile.txt", "w").close()
    f = open("moveLogFile.txt", 'a')
    f.write(str(pos))
    move = players.MCTS(env.pos, 300, "random", 0, True, 'average', None, 10000, None, False, 0, f)
    print("end reached")
    f.close()

# Initialize variables
env = DraughtsScape()
obs = env.reset()
Scan.set("variant", "normal")
Scan.set("book", "false")
Scan.set("book-ply", "4")
Scan.set("book-margin", "4")
Scan.set("ponder", "false")
Scan.set("threads", "1")
Scan.set("tt-size", "24")
Scan.set("bb-size", "6")
Scan.update()
Scan.init()

players = Players.Players()

def run():
    # Let the given player make the first 100 puzzles
    # checkPuzzles(0, 100, False, lambda: MCTSWieger.runPos(env.pos, 10000))

    # play a set of 10 games between MCTS with 10000 nodes and simple 5 deep minimax
    maxGames = 10
    playGame(lambda: players.MCTS(env.pos, 300, "random", 0, True, 'average', None, 10000, None, False, 0), lambda: players.shuflleMinimax(env.pos, 5),  maxGames, players)

    # Play self against the computer
    # maxGames = 1
    # playGame(lambda: players.human(env.pos, False), lambda: players.MCTS(env.pos, 1, "random", 0, True, 'average', None, None, None, False, 0), maxGames, players)

    # Train, test and evaluate a neural network for trap recognition
    # neuralNetworks = NeuralNetworks.NeuralNetworks()
    # model = neuralNetworks.createNeuralNetwork(True, 'trainingSets/trapTrainingImproved.csv')

    # playGame(lambda: players.MCTS(env.pos, 1, "random", 3, True, 'average', None, None, neuralNetworks, False, 0), lambda:  players.MCTS(env.pos, 1, "random", 3, True, 'average', None, None, neuralNetworks, False, 0),  maxGames, players)

    # print("truePositives: " + str(neuralNetworks.truePositive))
    # print("falsePositives: " + str(neuralNetworks.falsePositive))
    # print("trueNegatives: " + str(neuralNetworks.trueNegative))
    # print("falseNegatives: " + str(neuralNetworks.falseNegative))


if __name__ == '__main__':
    run()