#!/usr/bin/env python3
"""Compare native ANGLEBITS2PHASE with the exact-turn Python reference.

Builds float/double/long-double evaluators at O0/Ofast and mixed optimization,
checking stored results in all four rounding modes, with DAZ/FTZ on and off.
Run: python3 bra/test/anglebits2phase_numerical.py [--sanitize]
"""

import argparse
from fractions import Fraction
import itertools
import pathlib
import random
import subprocess
import sys
import tempfile

from anglebits2phase_reference import Limits, decode_chain, evaluate_turns


def decode_storage(hex_bytes, digits, min_exponent):
    """Decode IEEE storage independently of native floating arithmetic."""
    data = bytes.fromhex(hex_bytes)
    raw = int.from_bytes(data, sys.byteorder)
    if digits == 64 and min_exponent == -16381:  # x87, ignore ABI padding
        if sys.byteorder != 'little':
            raise RuntimeError('unsupported big-endian x87 storage')
        raw &= (1 << 80) - 1
        fraction_bits, exponent_bits, explicit = 63, 15, True
    else:
        fraction_bits = digits - 1
        exponent_bits = {24: 8, 53: 11, 113: 15}[digits]
        explicit = False
    significand_bits = fraction_bits + int(explicit)
    negative = bool(raw >> (significand_bits + exponent_bits) & 1)
    exponent = raw >> significand_bits & ((1 << exponent_bits) - 1)
    significand = raw & ((1 << significand_bits) - 1)
    if exponent == (1 << exponent_bits) - 1:
        raise AssertionError('nonfinite native result')
    if exponent and not explicit:
        significand += 1 << fraction_bits
    scale = (exponent - ((1 << (exponent_bits - 1)) - 1) if exponent else min_exponent - 1) - fraction_bits
    value = Fraction(significand) * Fraction(2) ** scale
    return (-value if negative else value), negative


def rounded_result(value, digits, min_exponent):
    negative = value < 0
    value = abs(value)
    if not value:
        return Fraction(0), False
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(2) ** exponent:
        exponent -= 1
    unit = Fraction(2) ** max(exponent - digits + 1, min_exponent - digits)
    quotient, remainder = divmod(value, unit)
    if remainder > unit / 2 or (remainder == unit / 2 and quotient % 2):
        quotient += 1
    result = quotient * unit
    return (-result if negative else result), negative


def cases(digits, min_exponent):
    chains = [(), ('INV',), ('POW', '0.5'), ('POW', '0.5', 'INV'),
              ('INV', 'POW', '0.5'), ('POW', '2', 'POW', '0.5'),
              ('POW', '4', 'INV', 'POW', '0.5'), ('POW', '-1.5'),
              ('POW', '9007199254740993'), ('POW', '9007199254740993.0'),
              ('POW', '0.1'), ('POW', '-0.0'), ('POW', '1e-4000'),
              ('POW', '5e-324', 'POW', '1.7976931348623157e308'),
              ('POW', '1.7976931348623157e308', 'POW', '0.5'),
              ('POW', '1.7976931348623158e308'),
              ('POW', f'{5 ** 1075}e-1075'),
              ('POW', f'{5 ** 1075 * 10 + 1}e-1076'),
              ('POW', f'{5 ** 1075 * 10 - 1}e-1076'),
              ('POW', '1.00000000000000011102230246251565404236316680908203125'),
              ('POW', '1.00000000000000033306690738754696212708950042724609375')]
    for width in range(1, 33):
        modulus = 1 << width
        for bits in sorted({0, 1, modulus // 2 - 1, modulus // 2, modulus // 2 + 1, modulus - 1}):
            if bits < modulus:
                for chain in chains:
                    yield bits, width, chain
    for width in range(1, 7):
        for bits in range(1 << width):
            for chain in chains[:8]:
                yield bits, width, chain
    rng = random.Random(396)
    for _ in range(100):
        width = rng.randrange(1, 33)
        chain = tuple(itertools.chain.from_iterable(
            ('INV',) if rng.randrange(3) == 0 else ('POW', rng.choice(['0.5', '-1.5', '0.1', '17', '-2', '0']))
            for _ in range(rng.randrange(1, 9))))
        yield rng.randrange(1 << width), width, chain
    # Roots below the backend turn grid can still yield nonzero radians.
    # Exercise smallest subnormals, signed zero, recovery by later factors,
    # and ties in the final result without first rounding turns to REAL.
    for depth in (-min_exponent + digits - 4, -min_exponent + digits,
                  -min_exponent + digits + 4, 50000):
        halves, remainder = divmod(depth, 1022)
        chain = ('POW', repr(2.0 ** -1022)) * halves
        if remainder:
            chain += ('POW', repr(2.0 ** -remainder))
        yield 1, 1, chain
        yield 1, 1, chain + ('INV',)
        yield 1, 1, chain + ('POW', '1.7976931348623157e308')
    yield 1, 1, ('INV',) * 1024
    yield 1, 1, ('POW', '9' * 4096)
    # The denominator reaches exactly the allowed size, then exceeds it.
    # Check the inclusive 65536-bit bound, and no later-zero shortcut.
    for count in (65534, 65535):
        chunks, remainder = divmod(count, 1022)
        chain = ('POW', repr(2.0 ** -1022)) * chunks
        if remainder:
            chain += ('POW', repr(2.0 ** -remainder))
        yield 1, 1, chain
        if count == 65535:
            yield 1, 1, chain + ('POW', '0')
            yield 0, 1, chain + ('POW', '0')
    for bits, width in ((0, 0), (0, 33), (2, 1)):
        yield bits, width, ('POW', '0')
    for chain in [('POW',), ('BAD',), ('POW', 'nan'), ('POW', '-inf'), ('POW', '1e99999'),
                  ('POW', '1.7976931348623159e308'),
                  ('POW', '0', 'POW', 'nan'), ('POW', 'A'), ('POW', ':PI'),
                  ('POW', '1_0'), ('POW', '0x1'), ('POW', '.'), ('POW', '1e'),
                  ('POW', '1e2e3'), ('POW', '1.2.3'), ('POW', '+' + '0' * 4096),
                  ('INV',) * 1025]:
        yield 1, 1, chain


def check(executable):
    # Read precision and stored TWO_PI without running any floating arithmetic
    # on native results (particularly DAZ-sensitive subnormals).
    metadata = subprocess.run([str(executable)], input='', text=True, capture_output=True, check=True).stdout.split()
    digits, min_exponent = map(int, metadata[:2])
    tau, _ = decode_storage(metadata[2], digits, min_exponent)
    samples = list(cases(digits, min_exponent))
    expected = []
    for bits, width, chain in samples:
        try:
            phase = evaluate_turns(bits, width, decode_chain(chain), Limits(uint_width=32))
            expected.append(rounded_result(phase * tau, digits, min_exponent))
        except ValueError:
            expected.append(None)
    for rounding, flush in itertools.product(range(4), range(2)):
        source = ''.join(f'{rounding} {flush} {bits} {width} {" ".join(chain)}\n' for bits, width, chain in samples)
        result = subprocess.run([str(executable)], input=source, text=True, capture_output=True, timeout=120, check=True)
        lines = result.stdout.splitlines()[1:]
        if len(lines) != len(samples):
            raise AssertionError(f'wrong result count: {len(lines)} != {len(samples)}')
        for sample, want, line in zip(samples, expected, lines):
            if want is None:
                if not line.startswith('ERR '):
                    raise AssertionError(f'{sample}: expected error, got {line}')
            elif not line.startswith('OK ') or decode_storage(line[3:], digits, min_exponent) != want:
                raise AssertionError(f'{sample}, rounding={rounding}, flush={flush}: expected {want}, got {line}')
    return len(samples) * 8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--sanitize', action='store_true')
    options = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    common = [options.cxx, '-std=c++14', '-Wall', '-Wextra', '-DBRA_NO_MPI',
              '-I' + str(root / 'bra/include'), '-I' + str(root / 'ket/include')]
    sanitizer = ['-fsanitize=undefined', '-fno-sanitize-recover=all'] if options.sanitize else []
    with tempfile.TemporaryDirectory(prefix='braket-anglephase-kernel-') as directory:
        build = pathlib.Path(directory)
        for precision, macro in (('float', 2), ('double', 1), ('long-double', 0)):
            flags = common + [f'-DBRA_REAL_TYPE={macro}'] + sanitizer
            probe = build / f'{precision}-probe.o'
            subprocess.run(flags + ['-O2', '-c', str(root / 'bra/test/anglebits2phase_numerical.cpp'), '-o', str(probe)], check=True)
            for mode in ('debug', 'release', 'mixed-release'):
                evaluator = build / f'{precision}-{mode}.o'
                executable = build / f'{precision}-{mode}'
                optimization = ['-Ofast', '-DNDEBUG'] if mode == 'release' else ['-O0']
                link = ['-O0'] if mode == 'debug' else ['-Ofast', '-DNDEBUG']
                subprocess.run(flags + optimization + ['-c', str(root / 'bra/src/classical_function.cpp'), '-o', str(evaluator)], check=True)
                subprocess.run([options.cxx, str(probe), str(evaluator), '-o', str(executable)] + link + sanitizer, check=True)
                count = check(executable)
                print(f'{precision} {mode}: {count} exact checks passed', flush=True)


if __name__ == '__main__':
    main()
