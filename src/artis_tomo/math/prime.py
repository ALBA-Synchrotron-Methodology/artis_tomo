"""
Prime numbers generator.

@author: joton
"""

from itertools import compress


def prime(n):
    """Return  a list of primes < n for n > 2."""
    sieve = bytearray([True]) * (n//2)

    for i in range(3, int(n**0.5)+1, 2):
        if sieve[i//2]:
            sieve[i*i//2::i] = bytearray((n-i*i-1)//(2*i)+1)
    return [2, *compress(range(3, n, 2), sieve[1:])]
