# coding=utf-8

from .misc import str_to_bool, consume

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
    def __init__(self, get_term_data):
        self.get_term_data = get_term_data

    def apply_action(self, action, operands, actions=None):
        if action == NEG:
            op = operands.pop()
            op.sign = False
            operands.append(op)

        if action == AND:
            op2 = operands.pop()
            op1 = operands.pop()

            if isinstance(op1.body, unicode):
                op1.body = self.get_term_data(op1.body)

            if isinstance(op2.body, unicode):
                op2.body = self.get_term_data(op2.body)

            op = None

            if op1.sign and op2.sign:
                op = Operand(op1.body & op2.body)
            elif not op1.sign:
                op = Operand(op2.body - op1.body)
            elif not op2.sign:
                op = Operand(op1.body - op2.body)
            elif not op1.sign and not op2.sign:
                op = Operand(op1.body & op2.body, False)

            operands.append(op)

        if action == OR:
            op2 = operands.pop()
            op1 = operands.pop()

            if isinstance(op1.body, unicode):
                op1.body = self.get_term_data(op1.body)

            if isinstance(op2.body, unicode):
                op2.body = self.get_term_data(op2.body)

            op = Operand(op1.body | op2.body)

            operands.append(op)

    def parse_query(self, query):
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
                op = Operand(curr_word.strip(), True)
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

        if isinstance(operands[1].body, unicode):
            return self.get_term_data(operands[1].body)
        return operands[1].body
