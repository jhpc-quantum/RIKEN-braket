#!/usr/bin/env python3
"""Execute whole-array to static-slice assignments."""

import argparse
import pathlib
import resource

from qasm2qcx_runtime_for_numerical import check_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra.resolve()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    check_program(bra, '''array[int,3] a={1,2,3}; a[2:-1:0]=a;''',
                  ('A1:0', 'A1:1', 'A1:2'), ['3', '2', '1'])


if __name__ == '__main__':
    main()
