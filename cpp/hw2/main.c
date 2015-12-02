#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define OPERATOR_ACTION 1
#define OPERAND_ACTION 0

#define STACK_SIZE 50

#define K_ADD '+'
#define K_SUB '-'
#define K_DIV '/'
#define K_MUL '*'
#define K_LBR '('
#define K_RBR ')'
#define K_SPACE ' '

typedef enum {
    EMP,
    LBR,
    ADD,
    SUB,
    MUL,
    DIV,
    UNA,
    RBR
} Operator;

void throw_error(const char *msg) {
    printf("%s", msg);
    exit(0);
}

typedef struct {
    unsigned char *data;
    int sp;
    int size;
} ActionStack;

ActionStack *newActionStack(int initial_size);
void deleteActionStack(ActionStack *stack);
unsigned char popActionStack(ActionStack *stack);
void pushActionStack(ActionStack *stack, unsigned char x);

typedef struct {
    Operator *data;
    int sp;
    int size;
} OperatorStack;

OperatorStack *newOperatorStack(int initial_size);
void deleteOperatorStack(OperatorStack *stack);
Operator popOperatorStack(OperatorStack *stack);
void pushOperatorStack(OperatorStack *stack, Operator x);

typedef struct {
    float *data;
    int sp;
    int size;
} OperandStack;

OperandStack *newOperandStack(int initial_size);
void deleteOperandStack(OperandStack *stack);
float popOperandStack(OperandStack *stack);
void pushOperandStack(OperandStack *stack, float x);

void applyOperator(OperandStack *stack, ActionStack *actionStack,
                   Operator anOperator) {

    if (anOperator == UNA) {
        float x = popOperandStack(stack);
        pushOperandStack(stack, -x);
    } else {

        float operand2 = popOperandStack(stack);
        float operand1 = popOperandStack(stack);

        if (anOperator == ADD) pushOperandStack(stack, operand1 + operand2);
        if (anOperator == SUB){
            pushOperandStack(stack, operand1 - operand2);
        }
        if (anOperator == DIV) {
            if (operand2 == 0) throw_error("[error]");
            pushOperandStack(stack, operand1 / operand2);
        }
        if (anOperator == MUL) pushOperandStack(stack, operand1 * operand2);
    }

    pushActionStack(actionStack, OPERAND_ACTION);
}

int main() {
    ActionStack *actionStack = newActionStack(STACK_SIZE);
    OperatorStack *operatorStack = newOperatorStack(STACK_SIZE);
    OperandStack *operandStack = newOperandStack(STACK_SIZE);

    pushOperatorStack(operatorStack, EMP);

    char c = EOF;
    while ((c = (char)getchar()) != '\n' && c != EOF) {
        float x;
        Operator currOperator = EMP;

        switch (c) {
            case K_SPACE: break;
            case K_SUB:
                // Is unary minus
                if (popActionStack(actionStack) &&
                    popActionStack(actionStack)) {
                    currOperator = UNA;
                } else {
                    currOperator = SUB;
                }
                break;
            case K_ADD: currOperator = ADD; break;
            case K_DIV: currOperator = DIV; break;
            case K_MUL: currOperator = MUL; break;
            case K_LBR: currOperator = LBR; break;
            case K_RBR: currOperator = RBR; break;
            default:
                ungetc(c, stdin);
                if (scanf("%f", &x) != 1) {
                    throw_error("[error]");
                    exit(0);
                } else {
                    pushOperandStack(operandStack, x);
                    pushActionStack(actionStack, OPERAND_ACTION);
                }
                continue;
        }

        if (currOperator != EMP) {
            // push LBR into operatorStack
            if (currOperator == LBR) {
                pushOperatorStack(operatorStack, currOperator);
                pushActionStack(actionStack, OPERATOR_ACTION);
                continue;
            }

            // evaluate expression inside brackets
            if (currOperator == RBR) {
                Operator operatorForApply;
                while ((operatorForApply = popOperatorStack(operatorStack)) !=
                       LBR) {
                    applyOperator(operandStack, actionStack, operatorForApply);
                }
                continue;
            }

            // Insert current operator
            Operator prevOperator = popOperatorStack(operatorStack);
            if ((int)prevOperator < (int)currOperator ) {
                pushOperatorStack(operatorStack, prevOperator);
            } else {
                applyOperator(operandStack, actionStack, prevOperator);
            }

            pushOperatorStack(operatorStack, currOperator);
            pushActionStack(actionStack, OPERATOR_ACTION);
        }
    }
    // Evaluate result
    Operator operator;
    while ((operator= popOperatorStack(operatorStack)) != EMP) {
        applyOperator(operandStack, actionStack, operator);
    }

    // Round and Print result
    printf("%.2f", roundf(popOperandStack(operandStack) * 100) / 100);

    // Clean up
    deleteActionStack(actionStack);
    deleteOperatorStack(operatorStack);
    deleteOperandStack(operandStack);

    return 0;
}

ActionStack *newActionStack(int initial_size) {
    ActionStack *stack = (ActionStack *)malloc(sizeof(ActionStack));
    stack->sp = 0;
    stack->size = initial_size;
    stack->data = (unsigned char *)malloc(sizeof(unsigned char) * stack->size);
    return stack;
}

void deleteActionStack(ActionStack *stack) {
    free(stack->data);
    free(stack);
}

unsigned char popActionStack(ActionStack *stack) {
    if (stack->sp == 0) return 1;
    return stack->data[--stack->sp];
}

void pushActionStack(ActionStack *stack, unsigned char x) {
    if (stack->size == stack->sp) {
        stack->size = (stack->size * 3 + 1) / 2;
        stack->data = (unsigned char *)realloc(
            stack->data, stack->size * sizeof(unsigned char));
    }
    stack->data[stack->sp++] = x;
}

OperatorStack *newOperatorStack(int initial_size) {
    OperatorStack *stack = (OperatorStack *)malloc(sizeof(OperatorStack));
    stack->sp = 0;
    stack->size = initial_size;
    stack->data = (Operator *)malloc(sizeof(Operator) * stack->size);
    return stack;
}

void deleteOperatorStack(OperatorStack *stack) {
    free(stack->data);
    free(stack);
}

Operator popOperatorStack(OperatorStack *stack) {
    if (stack->sp == 0) {
        throw_error("[error]");
        exit(0);
    }

    Operator x = stack->data[--stack->sp];
    return x;
}

void pushOperatorStack(OperatorStack *stack, Operator x) {
    if (stack->size == stack->sp) {
        stack->size = (stack->size * 3 + 1) / 2;
        stack->data =
            (Operator *)realloc(stack->data, stack->size * sizeof(Operator));
    }
    stack->data[stack->sp++] = x;
}

OperandStack *newOperandStack(int initial_size) {
    OperandStack *stack = (OperandStack *)malloc(sizeof(OperandStack));
    stack->sp = 0;
    stack->size = initial_size;
    stack->data = (float *)malloc(sizeof(float) * stack->size);
    return stack;
}

void deleteOperandStack(OperandStack *stack) {
    free(stack->data);
    free(stack);
}

float popOperandStack(OperandStack *stack) {
    if (stack->sp == 0) {
        throw_error("[error]");
        exit(0);
    }
    float x = stack->data[--stack->sp];
    return x;
}

void pushOperandStack(OperandStack *stack, float x) {
    if (stack->size == stack->sp) {
        stack->size = (stack->size * 3 + 1) / 2;
        stack->data =
            (float *)realloc(stack->data, stack->size * sizeof(float));
    }
    stack->data[stack->sp++] = x;
}
