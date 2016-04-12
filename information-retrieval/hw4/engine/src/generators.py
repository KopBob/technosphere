def gen_empty():
    return
    yield


def gen_trace(gen):
    for i, v in enumerate(gen):
        yield v
        if i < 10:
            print v


def gen_gen(iterator):
    for val in iterator:
        yield val


def gen_sync(gen, threshold):
    while 1:
        val = gen.next()
        if val >= threshold:
            yield val


def gen_with_end(gen):
    for val in gen:
        yield val
    yield None


def gen_cumdiff(gen):
    prev = 0
    for curr in gen:
        yield curr - prev
        prev = curr


def gen_cumsum(gen):
    prev = 0
    for curr in gen:
        prev += curr
        yield prev


def gen_intersect(gen1, gen2):
    val1 = gen1.next()
    val2 = gen2.next()
    while True:
        if val1 == val2:
            yield val1
            val1 = gen1.next()
            val2 = gen2.next()
        elif val1 > val2:
            val2 = gen_sync(gen2, val1).next()
        elif val1 < val2:
            val1 = gen_sync(gen1, val2).next()


def gen_union(gen1, gen2):
    gen1 = gen_with_end(gen1)
    gen2 = gen_with_end(gen2)

    val1 = gen1.next()
    val2 = gen2.next()

    while True:
        if val1 is None:
            while val2 is not None:
                yield val2
                val2 = gen2.next()
            break
        if val2 is None:
            while val1 is not None:
                yield val1
                val1 = gen1.next()
            break

        if val1 < val2:
            yield val1
            val1 = gen1.next()
        elif val1 == val2:
            yield val1
            val1 = gen1.next()
            val2 = gen2.next()
        elif val1 > val2:
            yield val2
            val2 = gen2.next()


def gen_subtract(gen1, gen2):
    gen2 = gen_with_end(gen2)

    val1 = gen1.next()
    val2 = gen2.next()

    while True:
        if val2 is None:
            while val1 is not None:
                yield val1
                val1 = gen1.next()

        if val1 == val2:
            val1 = gen1.next()
            val2 = gen2.next()
        elif val1 < val2:
            yield val1
            val1 = gen1.next()
        elif val1 > val2:
            val2 = gen2.next()


if __name__ == '__main__':
    # gen_intersect

    r1 = range(1, 20)
    r2 = range(8, 15)
    print list(gen_intersect(gen_gen(r1), gen_gen(r2))) == list(set(r1) & set(r2))

    r1 = []
    r2 = range(8, 15)
    print list(gen_intersect(gen_gen(r1), gen_gen(r2))) == list(set(r1) & set(r2))

    r1 = range(8, 15)
    r2 = range(100, 500)
    print list(gen_intersect(gen_gen(r1), gen_gen(r2))) == list(set(r1) & set(r2))

    # gen_union

    r1 = range(1, 20)
    r2 = range(8, 55)
    print list(gen_union(gen_gen(r1), gen_gen(r2))) == list(set(r1) | set(r2))

    r1 = range(1, 20)
    r2 = []
    print list(gen_union(gen_gen(r1), gen_gen(r2))) == list(set(r1) | set(r2))

    r1 = []
    r2 = []
    print list(gen_union(gen_gen(r1), gen_gen(r2))) == list(set(r1) | set(r2))

    # gen_subtract

    r1 = range(1, 20)
    r2 = []  # range(8, 15)
    print list(gen_subtract(gen_gen(r1), gen_gen(r2))) == list(set(r1) - set(r2))

    r1 = range(1, 20)
    r2 = range(8, 15)
    print list(gen_subtract((i for i in r1), (i for i in r2))) == list(set(r1) - set(r2))
