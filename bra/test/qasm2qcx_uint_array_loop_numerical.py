#!/usr/bin/env python3

"""Execute UINT array loops and verify width-preserving iterator semantics.

Example:
  python3 bra/test/qasm2qcx_uint_array_loop_numerical.py --bra bra/bin/bra
"""

import argparse
import pathlib
import resource

from qasm2qcx_integer_array_numerical import check_error
from qasm2qcx_runtime_for_numerical import check_program, qasm2qcx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bra', required=True, type=pathlib.Path)
    bra = parser.parse_args().bra
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    native = qasm2qcx.QASM2QCXConverter.QCX_UINT_WIDTH
    native_mask = qasm2qcx.QASM2QCXConverter.QCX_UINT_MAX

    document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
    marker = '<!-- uint-array-widening-example -->'
    if document.count(marker) != 1:
        raise RuntimeError('Expected exactly one UINT array widening example')
    example = document.split(marker, 1)[1].split('```qasm\n', 1)[1].split('```', 1)[0]
    check_program(bra, example.replace('OPENQASM 3.0;', '', 1),
                  ('COPIED63:0', 'COPIED63:1', 'TOTAL31', 'INDEX31'), ['3', '15', '20', '2'])

    # The copied value retains its number, while operations use iterator width.
    check_program(bra, '''array[uint[4],2] values={15,1}; uint total=0; uint complement=0;
        for uint[8] value in values { total+=value+uint[8](1); complement+=~value; }''',
        ('TOTAL31','COMPLEMENT1023'), ['18','494'])

    # Check every supported source/iterator width pair against Python integers.
    for source_width in range(1, native):
        maximum = (1 << source_width) - 1
        for target_width in range(source_width + 1, native + 1):
            mask = (1 << target_width) - 1
            iterators = (f'uint[{target_width}]', 'uint') if target_width == native else (f'uint[{target_width}]',)
            for iterator in iterators:
                check_program(bra, f'''array[uint[{source_width}],2] values={{0,{maximum}}};
                    array[uint[{target_width}],2] copied; array[uint[{target_width}],2] comp;
                    array[uint[{target_width}],2] wrapped; int i=0;
                    for {iterator} value in values {{ copied[i]=value; comp[i]=~value;
                        wrapped[i]=value+uint[{target_width}](-1); i+=1; }}''',
                    ('COPIED63:0','COPIED63:1','COMP15:0','COMP15:1',
                     'WRAPPED127:0','WRAPPED127:1','VALUES63:0','VALUES63:1','I1'),
                    list(map(str, (0, maximum, mask, mask ^ maximum, mask,
                                   (maximum - 1) & mask, 0, maximum, 2))))

    # Source changes affect future visits but do not change the copied value.
    check_program(bra, '''array[uint[4],3] values={1,2,3}; array[uint[8],3] copied;
        int i=0; for uint[8] value in values { copied[i]=value; values[i]=0; i+=1;
            if (i==1) { values[1]=7; continue; } if (i==2) { break; } }
        uint total=0; for uint[8] values in values { total+=values; }''',
        ('COPIED63:0','COPIED63:1','COPIED63:2','VALUES63:0','VALUES63:1',
         'VALUES63:2','TOTAL31','I1'), ['1','7','0','0','0','3','3','2'])

    check_program(bra, '''array[uint[4],2] values={1,2}; array[uint[2],1] other={3};
        uint total=0; uint[8] value=7; int bad=9;
        if (false) { for uint[8] v in values { total+=values[bad]; } }
        for int skip in [1:0] { for uint[8] v in values { total+=values[bad]; } }
        for uint[8] value in values { total+=value;
            for uint[8] value in other { total+=value; } total+=value; } total+=value;''',
        ('TOTAL31','VALUE31'), ['19','7'])

    for gate in ('rx(theta * pi)', 'ry(-float(theta) * pi)', 'U(theta * pi, 0, 0)'):
        check_program(bra, 'include "stdgates.inc"; qubit q; array[uint[4],2] values={0,1}; '
                      f'for uint[8] theta in values {{ {gate} q; }} '
                      'bit result=measure q; int n=int(result);', ('N1',), ['1'])

    for width in (1, 4, 8, native):
        mask = (1 << width) - 1
        for size in (1, 3):
            initial = [mask, 1, mask - 1][:size]
            for mode in ('all', 'continue', 'break'):
                values = initial.copy()
                total = complements = position = 0
                for index in range(size):
                    value = values[index]
                    position += 1
                    if mode == 'continue' and position == 2:
                        continue
                    if mode == 'break' and position == 3:
                        break
                    if position < size:
                        values[position] = (values[position] + 1) & mask
                    values[position - 1] = 0
                    total = (total + value) & native_mask
                    complements = (complements + (value ^ mask)) & native_mask
                transfer = ('if (i == 2) { continue; }' if mode == 'continue'
                            else 'if (i == 3) { break; }' if mode == 'break' else '')
                check_program(bra, f'''array[uint[{width}], {size}] values = {{{', '.join(map(str, initial))}}};
                    uint total = 0; uint complements = 0; int i = 0;
                    for uint[{width}] value in values {{ i += 1; {transfer}
                        if (i < {size}) {{ values[i] += uint[{width}](1); }}
                        values[i - 1] = 0; total += value; complements += ~value;
                    }}''', ('TOTAL31', 'COMPLEMENTS2047', 'I1') + tuple(f'VALUES63:{i}' for i in range(size)),
                              [str(total), str(complements), str(position)] + list(map(str, values)))

    # The source declaration and each nested iterator can have different widths.
    check_program(bra, '''uint[8] value = 17; array[uint[4], 1] small = {15};
        array[uint[8], 1] large = {255}; uint total = 0;
        for uint[4] value in small { total += value + uint[4](1);
            for uint[8] value in large { total += value + uint[8](1); }
            total += ~value;
        } total += value;''', ('TOTAL31', 'VALUE31'), ['17', '17'])
    check_program(bra, '''array[uint[4], 2] values = {1, 3}; uint total = 0;
        for uint[4] value in values { for uint[4] other in values { total += value * other; } }
        uint powers = 0; for uint[4] value in values { powers += value ** uint[4](2); }''',
                  ('TOTAL31', 'POWERS63'), ['16', '10'])

    for base, iterator in (('uint', f'uint[{native}]'), (f'uint[{native}]', 'uint')):
        check_program(bra, f'array[{base}, 2] values = {{0, 1}}; uint total = 0; '
                      f'for {iterator} value in values {{ total += value; }}', ('TOTAL31',), ['1'])

    for outer in ('for int outer in [0:2]', 'for int outer in [0:stop]',
                  'for int outer in {0, stop, 1}', 'while (visits < 3)'):
        check_program(bra, 'array[uint[4], 3] values = {1, 2, 3}; uint total = 0; int stop = 2; int visits = 0; '
                      + outer + ''' { visits += 1; for uint[4] value in values {
                          if (value == 3) { break; } total += value; } }''',
                      ('TOTAL31', 'VISITS63'), ['9', '3'])
    check_program(bra, '''array[uint[4], 2] values = {1, 2}; uint total = 0;
        for uint[4] value in values {
            for int j in [0:value] { total += j; }
            for int j in {value, 0} { total += j; }
        }''', ('TOTAL31',), ['7'])
    check_program(bra, '''array[uint[4], 2] values = {1, 2}; uint total = 0;
        for uint[4] values in values { total += values; } total += values[0];''',
                  ('TOTAL31',), ['4'])

    check_program(bra, '''array[uint[4], 2] values = {0, 1};
        array[float, 1] floats = {0.5}; bit[2] flags = "01"; float total = 0;
        for uint[4] value in values { for float value in floats { total += value; }
            for bit value in flags { total += int(value); }
            for int value in [0:1] { total += value; } total += float(value); }''',
                  ('TOTAL31',), ['6'])

    check_program(bra, '''array[uint[4], 2] values = {0, 1}; uint total = 0; bool flag = false;
        int bad = 2; int zero = 0;
        if (false) { for uint[4] value in values { total += values[bad]; } }
        for int outer in [1:0] { for uint[4] value in values { total += values[bad]; } }
        for uint[4] value in values { flag = true || values[bad] > value;
            flag = false && bool(values[1 / zero]); continue; total += values[bad]; }
        total += 1;''', ('TOTAL31', 'FLAG15'), ['1', '0'])

    for gate in ('rx(theta * pi)', 'ry(-float(theta) * pi)', 'U(theta * pi, 0, 0)'):
        check_program(bra, 'include "stdgates.inc"; qubit q; array[uint[4], 2] values = {0, 1}; '
                      f'for uint[4] theta in values {{ {gate} q; gphase(theta); }} '
                      'bit result = measure q; int n = int(result);', ('N1', 'VALUES63:1'), ['1', '1'])

    check_error(bra, '''array[uint, 1] values = {uint(-1)}; int total = 0;
        for uint value in values { for int j in [0:value] { total += j; } }''',
                'UINT to INT conversion out of range')
    check_error(bra, '''array[uint[4], 1] values = {4}; uint[4] target = 1;
        for uint[4] value in values { target <<= value; }''', 'assertion failed in ASSERT')
    check_error(bra, '''array[uint[4], 1] values = {1}; uint total = 0; int bad = 1;
        for uint[4] value in values { total += values[bad]; }''', 'assertion failed in ASSERT')

    print('UINT array loop numerical checks passed')


if __name__ == '__main__':
    main()
