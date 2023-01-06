#  (C) Copyright Wieger Wesselink 2022. Distributed under the GPL-3.0
#  Software License, (See accompanying file license.txt or copy at
#  https://www.gnu.org/licenses/gpl-3.0.txt)

import unittest
from draughts1 import *


class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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

    def test_position(self):
        pos = Pos()
        text = '..................................................W'
        self.assertEqual(text, print_position(pos, False, True))
        self.assertEqual(pos, parse_position(text))
        self.assertFalse(pos.can_move(Side.White))
        self.assertFalse(pos.can_capture(Side.White))
        self.assertFalse(pos.has_king())
        self.assertFalse(pos.is_threat())

        pos = start_position()
        text = 'xxxxxxxxxxxxxxxxxxxx..........ooooooooooooooooooooW'
        self.assertEqual(text, print_position(pos, False, True))
        self.assertEqual(pos, parse_position(text))
        self.assertTrue(pos.can_move(Side.White))
        self.assertFalse(pos.can_capture(Side.White))
        self.assertFalse(pos.has_king())
        self.assertFalse(pos.is_threat())
        moves = generate_moves(pos)
        self.assertEqual(9, len(moves))
        display_position(pos)
        for move in moves:
            print(print_move(move, pos))

        text = '''
           .   .   O   .   . 
         .   .   .   .   .   
           .   x   .   x   . 
         .   .   x   x   .   
           .   .   x   .   . 
         .   .   x   x   .   
           .   x   .   x   . 
         .   .   .   .   .   
           .   .   .   .   . 
         .   .   .   .   .   W;
        '''
        pos = parse_position(text)
        display_position(pos)
        print('eval', eval_position(pos))
        self.assertTrue(pos.can_move(Side.White))
        self.assertTrue(pos.can_capture(Side.White))
        self.assertTrue(pos.has_king())
        self.assertTrue(pos.has_king_side(Side.White))
        self.assertFalse(pos.is_threat())

        text = '''
           .   .   O   .   . 
         .   .   .   .   .   
           .   x   .   x   . 
         .   .   x   x   .   
           .   .   x   .   . 
         .   .   x   x   .   
           .   x   .   x   . 
         .   .   .   .   .   
           .   .   .   .   . 
         .   .   .   .   .   B;
        '''
        pos = parse_position(text)
        display_position(pos)
        print('eval', eval_position(pos))
        self.assertTrue(pos.can_move(Side.White))
        self.assertTrue(pos.can_move(Side.Black))
        self.assertTrue(pos.can_capture(Side.White))
        self.assertFalse(pos.can_capture(Side.Black))
        self.assertTrue(pos.has_king())
        self.assertTrue(pos.has_king_side(Side.White))
        self.assertFalse(pos.has_king_side(Side.Black))
        self.assertTrue(pos.is_threat())

    def test_parse(self):
        text = '..................................................B'
        pos = parse_position(text)
        self.assertEqual(text, print_position(pos, False, True))


    def test_search(self):
        text = '''
           .   .   .   .   . 
         .   .   .   .   .   
           .   .   x   x   . 
         x   x   x   x   .   
           x   .   x   x   o 
         x   o   o   .   o   
           .   o   o   .   o 
         .   o   o   o   .   
           .   .   .   .   . 
         .   .   .   .   .   B;
        '''
        pos = parse_position(text)
        display_position(pos)
        print('eval', eval_position(pos))
        print('hash', hash_key(pos), hash(pos))
        # N.B. The values are different, because the function hash truncates the return value to a size of Py_ssize_t.

        si = SearchInput()
        si.move = True
        si.book = False
        si.depth = 15
        si.nodes = 1000000000000
        si.time = 5.0
        si.input = True
        si.output = OutputType.Terminal

        so = SearchOutput()
        node = make_node(pos)
        search(so, node, si)
        self.test_egdb()

        # run_terminal_game()

    def test_egdb(self):
        text = '''
           .   .   .   .   X
         .   .   .   .   .
           .   .   .   .   .
         .   .   .   .   .
           .   .   .   .   .
         .   .   .   o   .
           .   .   .   .   .
         x   .   .   .   .
           .   .   .   .   O
         .   o   .   .   O   W;
        '''
        pos = parse_position(text)
        display_position(pos)
        value = EGDB.probe_raw(pos)  # N.B. probe_raw can not be called in capture positions
        print(value)
        testText = print_position(pos, False, True)
        print(testText)
        self.assertEqual(EGDBValue.Win, value)

        text = '''
           .   .   .   .   .
         .   .   .   .   X
           .   .   .   .   .
         .   .   .   .   .
           .   .   .   .   .
         .   .   .   o   .
           .   .   .   .   .
         x   .   .   .   .
           .   o   .   .   O
         .   .   .   .   O   B;
        '''
        pos = parse_position(text)
        display_position(pos)
        print("Really?")
        print(print_position(pos, False, True))
        value = EGDB.probe_raw(pos)
        print(value)
        self.assertEqual(EGDBValue.Loss, value)

        text = '''
           .   .   .   .   .
         .   .   .   .   .
           .   .   .   X   .
         .   .   .   .   .
           .   .   .   .   .
         .   .   .   o   .
           .   .   .   .   .
         x   .   .   .   .
           .   o   .   .   O
         .   .   .   .   O   B;
        '''
        pos = parse_position(text)
        display_position(pos)
        value = EGDB.probe_raw(pos)
        print("Value: " + str(value))
        print("Value: " + str(EGDB.probe(pos)))
        self.assertEqual(EGDBValue.Draw, value)

        text = '''
           .   .   .   .   .
         o   .   .   .   .
           .   .   .   .   .
         .   .   .   o   .
           .   .   .   .   .
         .   .   .   .   X
           .   .   .   .   x
         .   .   .   .   .
           .   .   .   .   o
         .   .   .   O   .   B;
        '''
        pos = parse_position(text)
        display_position(pos)
        value = EGDB.probe(pos)
        self.assertEqual(EGDBValue.Loss, value)

        text = '''
                   .   .   .   .   .
                 .   .   .   .   x
                   .   .   .   .   .
                 .   .   .   .   .
                   .   x   .   .   .
                 .   .   .   .   .
                   .   .   .   x   .
                 o   .   .   .   .
                   o   .   .   x   .
                 .   .   .   .   .   W;
                '''
        pos = parse_position(text)
        display_position(pos)
        print(EGDB.probe(pos))
        print(self.checkVicDB(pos))
        print("t")

        # self.assertEqual(EGDBValue.Loss, value)

    def checkVictory(self, pos):
        if not pos.can_move(Side.White):
            return -1
        elif not pos.can_move(Side.Black):
            return 1
        else:
            return None

    def checkVicDB(self, pos):
        posString = print_position(pos, False, True)
        numWhite = posString.lower().count("o")
        numBlack = posString.lower().count("x")
        if numWhite > 0 and numBlack > 0 and numWhite + numBlack <= 6:
            whitePlayer = False
            if print_position(pos, False, True)[-1] == "W":
                whitePlayer = True
            value = EGDB.probe(pos)
            if (whitePlayer and value == 2) or (not whitePlayer and value == 1):
                return 1
            elif (not whitePlayer and value == 2) or (whitePlayer and value == 1):
                return -1
            else:
                return 0
        else:
            return self.checkVictory(pos)


if __name__ == '__main__':
    import unittest
    unittest.main()
