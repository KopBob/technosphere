# coding=utf-8
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
                yield b

    def gen_decode(self, bytestream):
        return self.decode(bytestream)

    def decode(self, bytestream):
        n = 0
        for byte in bytestream:
            if byte < 128:
                n = n * 128 + byte
            else:
                n = n * 128 + (byte - 128)
                yield n

                n = 0
