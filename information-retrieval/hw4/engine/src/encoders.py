# coding=utf-8

import struct
import textwrap
import itertools


from .constants import simple9_table


class VarByte:
    def encode_number(self, n):
        bytes = []

        while (True):
            bytes.insert(0, n % 128)
            if n < 128:
                break
            n = n / 128
        bytes[-1] += 128

        return bytes

    def encode(self, numbers):
        bytestream = []

        for n in numbers:
            for b in self.encode_number(n):
                yield struct.pack('B', b)

    def gen_decode(self, bytestream):
        return self.decode(bytestream)

    def decode(self, bytestream):
        n = 0
        for byte in bytestream:
            byte = struct.unpack('B', byte)[0]
            if byte < 128:
                n = n * 128 + byte
            else:
                n = n * 128 + (byte - 128)
                yield n

                n = 0


class Simple9:
    def simple9_encode(self, scheme, arr):
        _, bits = simple9_table[scheme]
        s = format(scheme, '04b')
        for a in arr:
            s += format(a, '0%ib' % bits)
        return struct.pack('I', int(s, 2))

    def encode_partial(self, numbers):
        count = 0
        is_fitted = False

        for scheme, (amount, bits) in enumerate(simple9_table):
            k = amount
            for c in numbers:
                if len(bin(c)[2:]) <= bits:
                    k -= 1
                else:
                    break
                count += 1
                if k == 0:
                    is_fitted = True
                    break
            if is_fitted:
                return self.simple9_encode(scheme, numbers[:count]), numbers[count:]
            else:
                count = 0

    def encode(self, numbers):
        numbers = list(numbers)

        while True:
            bytes, numbers = self.encode_partial(numbers)
            for b in bytes:
                yield b
            if len(numbers) == 0:
                break

    def gen_decode(self, bytestream):
        return self.decode(bytestream)

    def decode(self, bytestream):
        while True:
            c = "".join(itertools.islice(bytestream, 0, 4))
            if len(c) == 0:
                break
            c = format(struct.unpack('I', c)[0], '032b')
            amount, bits = simple9_table[int(c[:4], 2)]
            parts = textwrap.wrap(c[4:], bits)
            assert amount == len(parts)
            for p in parts:
                yield int(p, 2)