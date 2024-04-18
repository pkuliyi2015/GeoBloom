'''
    Prepare Bloom Filter hash function
    This is the most naive implementation of a bloom filter in BloomNNUE.
'''


import math
import hashlib
import xxhash

from struct import pack, unpack


def bloom_param_estimate(capacity, error_rate=0.01):
    """Implements a space-efficient probabilistic data structure

    capacity
        this BloomFilter must be able to store at least *capacity* elements
        while maintaining no more than *error_rate* chance of false
        positives
    error_rate
        the error_rate of the filter returning false positives. This
        determines the filters capacity. Inserting more than capacity
        elements greatly increases the chance of false positives.
    """
    if not (0 < error_rate < 1):
        raise ValueError("Error_Rate must be between 0 and 1.")
    if not capacity > 0:
        raise ValueError("Capacity must be > 0")
    # given M = num_bits, k = num_slices, P = error_rate, n = capacity
    #       k = log2(1/P)
    # solving for m = bits_per_slice
    # n ~= M * ((ln(2) ** 2) / abs(ln(P)))
    # n ~= (k * m) * ((ln(2) ** 2) / abs(ln(P)))
    # m ~= n * abs(ln(P)) / (k * (ln(2) ** 2))
    num_slices = int(math.ceil(math.log(1.0 / error_rate, 2)))
    bits_per_slice = int(math.ceil(
        (capacity * abs(math.log(error_rate))) /
        (num_slices * (math.log(2) ** 2))))
    return num_slices, bits_per_slice


def make_hashfuncs(num_slices, num_bits):
    if num_bits >= (1 << 31):
        fmt_code, chunk_size = 'Q', 8
    elif num_bits >= (1 << 15):
        fmt_code, chunk_size = 'I', 4
    else:
        fmt_code, chunk_size = 'H', 2
    total_hash_bits = 8 * num_slices * chunk_size
    if total_hash_bits > 384:
        hashfn = hashlib.sha512
    elif total_hash_bits > 256:
        hashfn = hashlib.sha384
    elif total_hash_bits > 160:
        hashfn = hashlib.sha256
    elif total_hash_bits > 128:
        hashfn = hashlib.sha1
    else:
        hashfn = xxhash.xxh128

    fmt = fmt_code * (hashfn().digest_size // chunk_size)
    num_salts, extra = divmod(num_slices, len(fmt))
    if extra:
        num_salts += 1
    salts = tuple(hashfn(hashfn(pack('I', i)).digest()) for i in range(0, num_salts))

    def _hash_maker(key):
        if isinstance(key, str):
            key = key.encode('utf-8')
        else:
            key = str(key).encode('utf-8')

        i = 0
        for salt in salts:
            h = salt.copy()
            h.update(key)
            for uint in unpack(fmt, h.digest()):
                yield uint % num_bits
                i += 1
                if i >= num_slices:
                    return

    return _hash_maker, hashfn


# Test
if __name__ == '__main__':
    make_hashes, _ = make_hashfuncs(4, 8192)
    print(list(make_hashes('test')))