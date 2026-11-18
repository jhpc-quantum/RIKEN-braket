#!/usr/bin/env python3
"""Report exact coefficient growth for representative fixed-gate sequences."""

import argparse
import time

import qasm2qcx_exact_single_qubit_prototype as exact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-gates', type=int, default=2048)
    args = parser.parse_args()
    if args.max_gates < 1:
        parser.error('--max-gates must be positive')
    print('pattern gates numerator_bits denominator_exponent seconds')
    for pattern in (('t',), ('h', 't'), ('h', 't', 's', 'x')):
        matrix = exact.GATES['id']
        milestones = {1, args.max_gates}
        count = 2
        while count < args.max_gates:
            milestones.add(count)
            count *= 2
        started = time.perf_counter()
        for index in range(args.max_gates):
            matrix = exact.multiply_matrices(exact.GATES[pattern[index % len(pattern)]], matrix)
            count = index + 1
            if count in milestones:
                bits, denominator = exact.representation_size(matrix)
                print(f'{"-".join(pattern)} {count} {bits} {denominator} {time.perf_counter() - started:.6f}')


if __name__ == '__main__':
    main()
