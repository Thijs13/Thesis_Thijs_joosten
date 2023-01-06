import csv
from draughts1 import *

totMoves = 0.0
totPos = 0.0

getLoc='trainingSets/gameDataSet.csv'
with open(getLoc) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for count, row in enumerate(csv_reader):
        print(count)
        if not row[0][:4] == "Game":
            pos = parse_position(row[0])
            moves = generate_moves(pos)
            if len(moves) > 1:
                totMoves += len(moves)
                totPos += 1

print("totMoves: " + str(totMoves))
print("totMoves: " + str(totPos))
print("average: " + str(totMoves / totPos))

