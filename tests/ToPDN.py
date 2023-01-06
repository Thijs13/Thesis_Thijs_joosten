import math
from typing import List

from draughts1 import *

def print_pdn_moves(moves: List[str]):
    text = ''
    for i, m in enumerate(moves):
        if i % 2 == 0:
            text = text + '%3d' % (i / 2 + 1) + '.'
        text = text + m
        if i % 2 == 0:
            text = text + ' '
        if i % 10 == 9:
            text = text + '\n'
    return text


class PDNPrinter:
    def __init__(self):
        self.text = ''

    def write_tag(self, key, value):
        if value != "":
            self.text = self.text + '[%s "%s"]\n' % (key, value)

    def write_white(self, player):
        self.write_tag("White", player)

    def write_black(self, player):
        self.write_tag("Black", player)

    def write_event(self, event):
        self.write_tag("Event", event)

    def write_date(self, date):
        self.write_tag("Date", date)

    def write_site(self, site):
        self.write_tag("Site", site)

    def write_round(self, round):
        self.write_tag("Round", round)

    def write_result(self, result):
        self.write_tag("Result", result)

    def write_white_clock(self, clock):
        self.write_tag("WhiteClock", clock)

    def write_black_clock(self, clock):
        self.write_tag("BlackClock", clock)

    def write_moves(self, moves):
        self.text = self.text + print_pdn_moves(moves)

def to_pdn(white, black, result, moves) -> str:
    printer = PDNPrinter()
    printer.write_white(white)
    printer.write_black(black)
    printer.write_result(result)
    return printer.text + print_pdn_moves(moves)