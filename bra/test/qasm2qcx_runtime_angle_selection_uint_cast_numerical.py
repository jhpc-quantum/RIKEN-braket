#!/usr/bin/env python3
"""Audit UINT casts from runtime angle-bit indices and sets (#531).

Run with --bra /path/to/bra, including a non-MPI UBSan build. Independent
little-endian bit vectors supply expected values.
"""

import argparse
import contextlib
import pathlib
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_angle_bit_cast_numerical as backend
from qasm2qcx_runtime_range_equality_numerical import binary, unary
from qasm2qcx_runtime_range_angle_cast_numerical import packed


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    setUp = backend.BackendTests.setUp
    execute = backend.BackendTests.execute
    program = backend.BackendTests.program
    check_exact = backend.BackendTests.check_exact

    def test_all_widths_constants_arrays_and_exact_roundtrips(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        mask = (1 << native)-1
        for width in range(1, native+1):
            narrow = (1 << width)-1
            patterns = range(1 << width) if width <= 5 else sorted({0, narrow, 1 << (width-1),
                                                                    int(('10100110'*width)[:width], 2)})
            for pattern in patterns:
                source = pattern | (mask ^ narrow)
                for reverse in (False, True):
                    positions = list(range(width-1, -1, -1) if reverse else range(width))
                    tokens = ','.join('first' if i == 0 else str(pos) for i, pos in enumerate(positions))
                    bits = [(source >> pos)&1 for pos in positions]
                    self.check_exact(f'const angle[{native}] C=angle[{native}]("{source:0{native}b}"); '
                        f'const angle[{native}] D=angle[{native}]("{source^mask:0{native}b}"); '
                        f'angle[{native}] phase=C; array[angle[{native}],2] phases={{C,D}}; '
                        f'int first={positions[0]}; uint slot=1; '
                        f'uint[{width}] value=uint[{width}](phase[{{{tokens}}}]); '
                        f'uint[{width}] constant=uint[{width}](C[{{{tokens}}}]); '
                        f'uint[{width}] element=uint[{width}](phases[slot][{{{tokens}}}]); '
                        f'bit[{width}] back=bit[{width}](uint[{width}](phase[{{{tokens}}}])); '
                        'bool stable=phase==C && phases[slot]==D;',
                        (('value', None), ('constant', None), ('element', None), ('back', width), ('stable', None)),
                        (packed(bits), packed(bits), packed([1-bit for bit in bits]), packed(bits), 1))

    def test_single_index_unsized_target_and_repeated_aliases(self):
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for pattern in range(16):
            for index in (-4, -1, 0, 1, 2, 3):
                bit = (pattern >> (index % 4))&1
                self.check_exact(f'angle[4] phase=angle[4]("{pattern:04b}"); int index={index}; '
                    'uint[1] one=uint[1](phase[index]); uint[1] set=uint[1](phase[{index}]); '
                    'uint[4] repeated=uint[4](phase[{index,index,0,-4}]);',
                    (('one', None), ('set', None), ('repeated', None)),
                    (bit, bit, bit+(bit<<1)+((pattern&1)<<2)+((pattern&1)<<3)))
        for bit in (0, 1):
            indices = ','.join(['index']*native)
            self.check_exact(f'angle[1] phase=angle[1]("{bit}"); int index=0; '
                f'uint value=uint(phase[{{{indices}}}]); bit[{native}] back=bit[{native}](value);',
                (('value', None), ('back', native)), (((1<<native)-1)*bit,)*2)

    def test_nested_register_operations_and_unsigned_consumers(self):
        native = self.module.QASM2QCXConverter
        for pattern in range(16):
            lhs = ('phase[{first,1,2,3}]', [(pattern>>i)&1 for i in range(4)])
            rhs = ('wide[last:-1:first]', [((pattern^15)>>i)&1 for i in (3,2,1,0)])
            for count in (0, 1, 4, 5, native.QCX_UINT_MAX):
                turn = min(count, native.QCX_INT_MAX)
                nodes = [binary(lhs, rhs, '&'), binary(rhs, lhs, '|'), binary(lhs, rhs, '^'),
                         unary(binary(unary(lhs, '~'), rhs, '^'), '<<', count, 'distance'),
                         unary(binary(lhs, rhs, '^'), 'rotl', turn, 'turn')]
                source = (f'angle[4] phase=angle[4]("{pattern:04b}"); bit[129] wide="{pattern^15:0129b}"; '
                          f'int first=0; int last=3; uint distance={count}; uint turn={turn}; ')
                for index, (expression, _) in enumerate(nodes):
                    source += f'uint[4] a{index}=uint[4]({expression}); '
                source += 'bool truth=bool(uint[4](phase[{first,1,2,3}])); uint total=popcount(bit[4](uint[4](phase[{first,1,2,3}])));'
                self.check_exact(source, tuple((f'a{i}', None) for i in range(len(nodes)))+(('truth', None), ('total', None)),
                                 tuple(packed(node[1]) for node in nodes)+(int(bool(pattern)), pattern.bit_count()))

    def test_array_iterators_fresh_values_and_overlap(self):
        for pattern in range(16):
            reverse = int(f'{pattern:04b}'[::-1], 2)
            self.check_exact(f'angle[4] phase=angle[4]("{pattern:04b}"); '
                'array[angle[4],2] phases={0.0,phase}; int slot=1; int first=0; int last=3; '
                'uint[4] saved=uint[4](phase[{first,1,2,last}]); '
                'phase=angle[4](bit[4](uint[4](phase[{last,2,1,first}]))); '
                'phases[slot]=angle[4](bit[4](uint[4](phases[slot][{last,2,1,first}]))); '
                'uint[4] element=uint[4](phases[slot][{first,1,2,last}]);',
                (('saved', None), ('phase', None), ('element', None)), (pattern, reverse, reverse))
        self.check_exact('array[angle[4],2] phases={angle[4]("1011"),angle[4]("1101")}; '
            'int first=0; uint total=0; for angle[4] phase in phases { total+=uint[4](phase[{first,1,2,3}]); } '
            'int slot=0; uint[4] saved; while(slot<2) { saved=uint[4](phases[slot][{first,1,2,3}]); slot+=1; }',
            (('total', None), ('saved', None), ('slot', None)), (24, 13, 2))

    def test_skipped_domains_and_reached_failures(self):
        calls = ('value=uint[4](phase[{index,999,-999,index}]);',
                 'value=uint[4](phases[1/zero][{index,0,1,2}]);',
                 'value=uint[4](rotl(phase[{index,0,1,2}],1/zero));')
        wrappers = ('if(false) {{ {} }}', 'while(false) {{ {} }}', 'for int i in [1:0] {{ {} }}',
                    'for int i in [low:high] {{ {} }}', 'while(true) {{ break; {} }}')
        for call in calls:
            for wrapper in wrappers:
                self.check_exact('angle[4] phase=pi; array[angle[4],2] phases={pi,0.0}; uint[4] value=6; '
                    'int index=999; int zero=0; int low=2; int high=1; '+wrapper.format(call),
                    (('value', None), ('phase', None)), (6, 8))
        self.check_exact('angle[4] phase=pi; int index=999; int zero=0; '
            'bool no=false&&bool(uint[1](phase[index])); '
            'bool yes=true||bool(uint[4](rotl(phase[{index,0,1,2}],1/zero)));',
            (('no', None), ('yes', None)), (0, 1))
        cases = (('phase[{index,0,1,99}]', 'assertion failed'),
                 ('phase[{index,0,1/zero,3}]', 'division by zero'),
                 ('phases[1/zero][{index,0,1,99}]', 'division by zero'),
                 ('phase[{index,0,1,2}]<<distance', 'assertion failed'),
                 ('rotl(phase[{index,0,1,2}],huge)', 'out of range'))
        for constant in (False, True):
            for expression, error in cases:
                source = (('const ' if constant else '')+'angle[4] phase=angle[4]("1011"); '
                    'array[angle[4],2] phases={phase,phase}; int index=0; int zero=0; int distance=-1; '
                    f'uint huge={self.module.QASM2QCXConverter.QCX_UINT_MAX}; uint[4] value; '
                    f'value=uint[4]({expression});')
                program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
                c = self.module.QASM2QCXConverter(program); c.visit(program)
                self.assertEqual(c._QASM2QCXConverter__used_temporary_variables, set())
                self.assertEqual(c._QASM2QCXConverter__captured_bit_register_values, {})
                name = c._QASM2QCXConverter__capitalize_variable_name('value')
                lines = []
                for line in c:
                    if line.startswith(f'LET {name} := QASM2QCX_'):
                        lines.append('PRINTLN 99999')
                    lines.append(line)
                self.assertIn('PRINTLN 99999', lines)
                result = self.execute(lines)
                self.assertNotEqual(result.returncode, 0, source)
                self.assertIn(error, result.stderr, source)
                self.assertNotIn('runtime error:', result.stderr)
                self.assertEqual(result.stdout, '')
                type(self).checked += 1

    def test_failed_indices_precede_extraction_and_later_operands(self):
        for expression, error in (('left[{index,0,1,99}] ^ right[{index,1/zero,2,3}]', 'assertion failed'),
                                  ('left[{index,0,1,2}] ^ right[{index,1/zero,2,3}]', 'division by zero')):
            source = ('angle[4] left=pi; angle[4] right=pi/2; int index=0; int zero=0; '
                      f'uint[4] value; value=uint[4]({expression});')
            program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
            c = self.module.QASM2QCXConverter(program)
            original = c._QASM2QCXConverter__runtime_angle_bit_value
            @contextlib.contextmanager
            def marked(selection, **kwargs):
                marker = {'left': 20000, 'right': 30000}[selection.source.name]
                lines = c._QASM2QCXConverter__qcx_lines
                lines.append(f'PRINTLN {marker+1}'); start = len(lines)
                with original(selection, **kwargs) as bits:
                    region = lines[start:]; instrumented = []
                    for index, line in enumerate(region):
                        if index and region[index-1].endswith(' &= 1') and ' := :INT:' in line:
                            instrumented.append(f'PRINTLN {marker+2}')
                        instrumented.append(line)
                    lines[start:] = instrumented
                    yield bits
            with patch.object(c, '_QASM2QCXConverter__runtime_angle_bit_value', side_effect=marked):
                c.visit(program)
            name = c._QASM2QCXConverter__capitalize_variable_name('value'); lines = []
            for line in c:
                if line.startswith(f'LET {name} := QASM2QCX_'):
                    lines.append('PRINTLN 99999')
                lines.append(line)
            self.assertIn('PRINTLN 99999', lines)
            result = self.execute(lines)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertNotIn('runtime error:', result.stderr)
            self.assertEqual(result.stdout.split(), ['20001'] if error == 'assertion failed'
                             else ['20001']+['20002']*4+['30001'])
            type(self).checked += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra', type=pathlib.Path)
    arguments = parser.parse_args()
    BackendTests.bra = arguments.bra.resolve() if arguments.bra else None
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime angle-bit selection UINT cast programs checked: {BackendTests.checked}')
    raise SystemExit(not result.wasSuccessful())
