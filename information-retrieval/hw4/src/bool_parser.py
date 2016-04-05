from src.misc import str_to_bool, consume

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


class BoolQueryParser:
    @staticmethod
    def apply_action(action, operands):
        if action == NEG:
            op1 = operands.pop()

            _str_act = "!(%s)" % op1
            print _str_act
            operands.append(_str_act)

            # operands.append(not str_to_bool(op1))

            return True

        if action == AND:
            op2 = operands.pop()
            op1 = operands.pop()

            _str_act = "%s & %s" % (op1, op2)
            print _str_act
            operands.append(_str_act)

            # operands.append(str_to_bool(op1) & str_to_bool(op2))

            return True

        if action == OR:
            op2 = operands.pop()
            op1 = operands.pop()

            _str_act = "%s | %s" % (op1, op2)
            print _str_act
            operands.append(_str_act)

            # operands.append(str_to_bool(op1) | str_to_bool(op2))

            return True

        return False

    @staticmethod
    def parse_query(query):
        operands = [EMP]
        actions = [EMP]

        query = query + ' '  # FixIt

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
                operands.append(curr_word)
                curr_word = ""

            if curr_act:
                if curr_act == LBR:
                    actions.append(curr_act)
                    continue

                if curr_act == RBR:
                    act = actions.pop()
                    while act != LBR:
                        BoolQueryParser.apply_action(act, operands)
                        act = actions.pop()
                    continue

                prev_act = actions.pop()
                if PRIORITY[prev_act] > PRIORITY[curr_act]:
                    BoolQueryParser.apply_action(prev_act, operands)
                else:
                    actions.append(prev_act)

                actions.append(curr_act)

        for act in reversed(actions):
            BoolQueryParser.apply_action(act, operands)
