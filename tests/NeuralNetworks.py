import copy
import math

from draughts1 import *
import csv
import random
# import keras
from keras.models import Sequential
import keras.layers as layers
from tensorflow import keras
import tensorflow as tf
import numpy as np

class NeuralNetworks:

    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0
    NNModel = None
    numTraps = 0
    numNormal = 0
    valueModel = None
    movesCor = 0
    movesWrong = 0
    valueDif = 0

    def posToList(self, pos):
        posText = print_position(pos, False, True)
        posList = []
        for char in posText:
            if char == ".":
                posList.append(0)
            elif char == "o":
                posList.append(1)
            elif char == "O":
                posList.append(3)
            elif char == "x":
                posList.append(-1)
            elif char == "X":
                posList.append(-3)
        return posList

    def getTrainingSetFromBoard(self, pos):
        tempList = self.posToList(pos)
        eval = eval_position(pos)
        if eval >= 0:
            tempList.append(1)
        else:
            tempList.append(0)
        f = open('trainingSets/test.csv', 'a', newline='')
        writer = csv.writer(f)
        writer.writerow(tempList)
        f.close()

    def checkTrap(self, pos):
        eval = piece_count_eval(pos)
        white = pos.is_white_to_move()
        for move in generate_moves(pos):
            newPos = pos.succ(move)
            newMoves = generate_moves(newPos)
            storeMove = move
            if len(newMoves) == 1:
                newPos = newPos.succ(newMoves[0])
                if (white and eval - piece_count_eval(newPos) >= 1) or (not white and eval - piece_count_eval(newPos) <= -1):
                    for move in generate_moves(newPos):
                        finalPos = play_forced_moves(newPos.succ(move))
                        finalEval = piece_count_eval(finalPos)
                        if (white and finalEval - eval >= 1) or (not white and finalEval - eval <= -1):
                            # print("new trap")
                            # print(print_move(storeMove, pos))
                            # print(eval)
                            # display_position(pos)
                            # print(finalEval)
                            # display_position(finalPos)
                            return True
        return False

    def getTrapTrainingSet(self, pos, writer = None):
        for move in generate_moves(pos):
            newPos = pos.succ(move)
            if self.checkTrap(newPos):
                tempList = self.posToList(newPos)
                tempList.append(1)
                self.numTraps += 1
            else:
                if self.numNormal <= self.numTraps and random.randint(0, 10) == 1:
                    tempList = self.posToList(newPos)
                    tempList.append(0)
                    self.numNormal += 1
                else:
                    return
            writer.writerow(tempList)

    def getValueTrainingSet(self, players, getLoc='trainingSets/gameDataSet.csv', setLoc='trainingSets/ValueTrainingSet.csv'):
        with open(getLoc) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for count, row in enumerate(csv_reader):
                print(count)
                if not row[0][:4] == "Game":
                    pos = parse_position(row[0])
                    tempList = self.posToList(pos)
                    bestValue, bestMove = players.alphaBeta(pos, 5, pos.is_white_to_move(), -math.inf, math.inf)
                    tempList.append(bestValue)
                    f = open(setLoc, 'a', newline='')
                    writer = csv.writer(f)
                    writer.writerow(tempList)
                    f.close()

    def createNeuralNetwork(self, conv=False, getLoc='trainingSets/augTrapTrainingSet.csv'):
        # load the dataset
        dataset = np.loadtxt(getLoc, delimiter=',', max_rows = 50000)
        # split into input (X) and output (y) variables
        X = dataset[:, 0:50]
        y = dataset[:, 50]

        if conv:
            model = Sequential()
            model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(50, 1)))
            # model.add(layers.MaxPooling1D(2))
            model.add(layers.Conv1D(64, kernel_size=3, activation='relu'))
            # model.add(layers.MaxPooling1D(2))
            model.add(layers.Conv1D(64, kernel_size=3, activation='relu'))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(1))
        else:
            model = Sequential()
            model.add(layers.Dense(24, input_dim=50, activation='relu'))
            model.add(layers.Dense(16, activation='relu'))
            model.add(layers.Dense(16, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(X, y, epochs=200, batch_size=200)

        _, accuracy = model.evaluate(X, y)
        print('Accuracy: %.2f' % (accuracy * 100))
        self.NNModel = model
        return model

    def createValueNetwork(self, getLoc='trainingSets/valueTrainingSet.csv'):
        # load the dataset
        dataset = np.loadtxt(getLoc, delimiter=',')
        # split into input (X) and output (y) variables
        X = dataset[:, 0:50]
        y = dataset[:, 50]

        model = Sequential()
        model.add(layers.Dense(24, input_dim=50, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

        model.fit(X, y, epochs=150, batch_size=200)
        _, accuracy = model.evaluate(X, y)
        print('Accuracy: %.2f' % (accuracy * 100))
        self.valueModel = model
        return model

    def createTuytNetwork(self, getLoc='trainingSets/newTuytTrainingSet.csv'):
        # load the dataset
        dataset = np.loadtxt(getLoc, delimiter=',')
        # split into input (X) and output (y) variables
        X = dataset[:, 0:100]
        y = dataset[:, 100]

        model = Sequential()
        model.add(layers.Dense(256, input_dim=100, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(3, activation='sigmoid'))

        lr_schedule = tf.optimizers.schedules.ExponentialDecay(initial_learning_rate=.01, decay_steps=5000,
                                                               decay_rate=0.96)
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics = ["accuracy"])

        positionLabels = keras.utils.to_categorical(y, 3)

        model.fit(X, positionLabels, epochs=300, batch_size=200)
        _, accuracy = model.evaluate(X, positionLabels)
        print('Accuracy: %.2f' % (accuracy * 100))
        self.valueModel = model
        return model

    def updateModel(self, model, X, y):
        model.fit(X, y, epochs=1)
        return model

    def testNeuralNetwork(self, pos):
        posList = self.posToList(pos)
        posList = [posList]
        testInput = np.array(posList)
        # prediction = model.predict(testInput)
        prediction = (self.NNModel.predict(testInput) > 0.5).astype(int)
        trapExist = self.checkTrap(pos)
        if prediction[0] == 1 and trapExist:
            self.truePositive += 1
            print('truePositive')
        elif prediction[0] == 0 and not trapExist:
            self.trueNegative += 1
            print('trueNegative')
        elif prediction[0] == 1 and not trapExist:
            self.falsePositive += 1
            print('falsePositive')
        elif prediction[0] == 0 and trapExist:
            self.falseNegative += 1
            print('falseNegative')
        display_position(pos)

    def testValueNetwork(self, pos, corValue, corMove, tuyt = False):
        funcPos = pos
        posList = self.posToList(funcPos)
        tuytList = self.changeRowToTuyt(posList)
        # tuytList = [tuytList]
        if tuyt:
            testInput = np.array([tuytList])
            prediction = self.valueModel.predict(testInput)
            result = np.argmax(prediction)
            if corValue > 10 and result == 0:
                self.movesCor += 1
            elif corValue < -10 and result == 1:
                self.movesCor += 1
            elif (corValue < 10 and corValue > -10) and result == 2:
                self.movesCor += 1
            else:
                self.movesWrong += 1
        else:
            testInput = np.array([posList])
            prediction = self.valueModel.predict(testInput)
            bestMove = None
            bestValue = 0
            moves = generate_moves(funcPos)
            if len(moves) == 1:
                return
            for move in moves:
                newPos = funcPos.succ(move)
                posList = self.posToList(newPos)
                testInput = np.array([posList])
                # posList = [posList]
                # testInput = np.array(posList)
                prediction = self.valueModel.predict(testInput)
                print(prediction)
                if prediction > bestValue:
                    bestValue = prediction
                    bestMove = move
            if bestMove == corMove:
                self.movesCor += 1
            else:
                self.movesWrong += 1
            self.valueDif += abs(corValue - prediction)

    def playPdnGame(self, game, gameNumber, recordGameData = False):
        pos = start_position()
        location = "trainingSets/trapTrainingImproved.csv"
        f = open(location, 'a', newline='')
        writer = csv.writer(f)
        for move in game.moves:
            pos = pos.succ(move)
            for newMove in generate_moves(pos):
                newPos = pos.succ(newMove)
                self.getTrapTrainingSet(newPos, writer)
        f.close()
        pos = start_position()
        if recordGameData:
            GDSLoc = open('trainingSets/gameDataSet.csv', 'a', newline='')
            writer = csv.writer(GDSLoc)
            writer.writerow(["Game " + str(gameNumber)])
            writer.writerow([print_position(pos, False, True)])
            for move in game.moves:
                pos = pos.succ(move)
                writer.writerow([print_position(pos, False, True)])
            GDSLoc.close()

    def playPdnGames(self, location, startNum, endNum, recordGameData = False):
        games = parse_pdn_file(location)
        if endNum >= len(games):
            print("There are only " + str(len(games)) + " games")
            return
        for i in range(startNum, endNum):
            game = games[i]
            print("Game: " + str(i))
            self.playPdnGame(game, i, recordGameData)

    def augmentData(self, getLoc, postLoc):
        with open(getLoc) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                mirrorRow = []
                for i in range(50):
                    mirrorRow.append(int(row[49-i]) * -1)
                mirrorRow.append(row[50])
                f = open(postLoc, 'a', newline='')
                writer = csv.writer(f)
                writer.writerow(row)
                writer.writerow(mirrorRow)
                f.close()

    def changeRowToTuyt(self, row):
        whiteMen = []
        blackMen = []
        for piece in row:
            if int(piece) == 1:
                whiteMen.append(1)
                blackMen.append(0)
            elif int(piece) == -1:
                whiteMen.append(0)
                blackMen.append(1)
            else:
                whiteMen.append(0)
                blackMen.append(0)
        allMen = whiteMen
        allMen.extend(blackMen)
        return allMen

    def changeDataToTuyt(self, getLoc, postLoc):
        with open(getLoc) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                newLine = self.changeRowToTuyt(row[0:50])
                if int(row[50]) > 10:
                    newLine.append(0)
                elif int(row[50]) < -10:
                    newLine.append(1)
                else:
                    newLine.append(2)
                f = open(postLoc, 'a', newline='')
                writer = csv.writer(f)
                writer.writerow(newLine)
                f.close()