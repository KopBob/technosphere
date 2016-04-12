# coding=utf-8
import re

from .misc import str_to_bool, consume
from .generators import gen_subtract, gen_intersect, gen_union


SP = ' '
EMP = None
NEG = '!'
AND = '&'
OR = '|'
LBR = '('
RBR = ')'

PRIORITY = {
    EMP: 10,
    NEG: 2,
    AND: 1,
    OR: 0,
    LBR: -1
}

NONTERM = [SP, NEG, AND, OR, LBR, RBR, EMP]


class Operand:
    def __init__(self, body, sign=True):
        self.body = body
        self.sign = sign


class BoolQueryParser:
    def apply_action(self, action, operands, actions=None):
        if action == NEG:
            op = operands.pop()
            op.sign = False
            operands.append(op)

        if action == AND:
            op2 = operands.pop()
            op1 = operands.pop()

            op = None

            if op1.sign and op2.sign:
                op = Operand(gen_intersect(op1.body, op2.body))
            elif not op1.sign:
                op = Operand(gen_subtract(op2.body, op1.body))
            elif not op2.sign:
                op = Operand(gen_subtract(op1.body, op2.body))
            elif not op1.sign and not op2.sign:
                op = Operand(gen_intersect(op1.body, op2.body), False)

            operands.append(op)

        if action == OR:
            op2 = operands.pop()
            op1 = operands.pop()

            op = Operand(gen_union(op1.body, op2.body))

            operands.append(op)

    def parse_query(self, query, terms_gens):
        operands = [EMP]
        actions = [EMP]

        query = query.lower() + ' '  # FixIt

        curr_word = ""

        iterator = range(0, len(query)).__iter__()
        for i in iterator:
            c = query[i]
            curr_act = None

            if c == SP:
                continue
            elif c == NEG:
                curr_act = NEG
            elif c == AND:
                curr_act = AND
            elif c == OR:
                curr_act = OR
            elif c == LBR:
                curr_act = LBR
            elif c == RBR:
                curr_act = RBR
            else:
                k = 0
                while query[i + k] not in NONTERM:
                    curr_word += query[i + k]
                    k += 1

                consume(iterator, k - 1)
                try:
                    op = Operand(terms_gens[curr_word.strip()], True)
                except KeyError:
                    print curr_word.strip()
                    for key in terms_gens.keys():
                        print key,
                    raise KeyError
                operands.append(op)
                curr_word = ""

            if curr_act:
                if curr_act == LBR:
                    actions.append(curr_act)
                    continue

                if curr_act == RBR:
                    act = actions.pop()
                    while act != LBR:
                        self.apply_action(act, operands)
                        act = actions.pop()
                    continue

                prev_act = actions.pop()
                if PRIORITY[prev_act] >= PRIORITY[curr_act]:
                    self.apply_action(prev_act, operands)
                else:
                    actions.append(prev_act)

                actions.append(curr_act)

        for act in reversed(actions):
            self.apply_action(act, operands)

        return operands[1].body
