#!/usr/bin/env python3
"""Audit bitwise compounds with nested runtime-range RHS expressions (#523).

Run with --bra /path/to/bra (a non-MPI build, including UBSan builds).
Independent little-endian bit lists model original LHS reads, RHS operations,
and selected writes. Failure markers verify snapshot and validation order.
"""

import argparse
import cmath
import math
import operator
import pathlib
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_runtime_range_equality_numerical as registers
import qasm2qcx_runtime_set_expression_assignment_numerical as index_sets
from qasm2qcx_static_selection_expression_assignment_numerical import write_selected


OPERATIONS = (('&=',operator.and_),('|=',operator.or_),('^=',operator.xor))


def compound(before, positions, bits, function):
    return write_selected(before,positions,[function(before[position],bit) for position,bit in zip(positions,bits)])


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    assignments = 0
    setUp = registers.BackendTests.setUp
    execute = registers.BackendTests.execute
    program = registers.BackendTests.program
    check_exact = registers.BackendTests.check_exact

    def test_documented_runtime_expression_bitwise_compound_example(self):
        document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
        marker = '<!-- runtime-expression-bitwise-compound-example -->'
        source = document.split(marker,1)[1].split('```qasm\n',1)[1].split('```',1)[0]
        source = source.replace('OPENQASM 3.0;','',1)
        self.check_exact(source,(('result',4),('flags',8),('a',4),('one',1),('captured',4),
                                 ('visits',None),('static_ok',None),('set_ok',None),('range_ok',None),('untouched',None)),
                         (14,212,5,0,3,2,1,1,1,1))

    # Source patterns and RHS references are independent of destination handling.
    check_case = index_sets.BackendTests.check_case
    test_exhaustive_small_patterns_and_source_kinds = index_sets.BackendTests.test_exhaustive_small_patterns_and_source_kinds
    test_native_widths_wide_sources_and_negative_ranges = index_sets.BackendTests.test_native_widths_wide_sources_and_negative_ranges
    test_deep_trees_and_native_endpoint_distances = index_sets.BackendTests.test_deep_trees_and_native_endpoint_distances

    def check_nodes(self, source, nodes, before, preservation=()):
        width = len(nodes[0][1])
        size = len(before)
        self.assertEqual(size,2*width+1)
        odd = list(range(1,size,2))
        reordered = odd[1:]+odd[:1]
        static_set = '{'+','.join(str(position-size) for position in reordered)+'}'
        runtime_set = '{index'+''.join(f',{position}' for position in reordered[1:-1])
        runtime_set += ',tail}' if width>1 else '}'
        declarations = (f'int index={reordered[0]-size}; uint tail={reordered[-1]}; '
                        f'uint high={size-1}; int stride=-2; uint low=1;')
        selections = (('',list(range(width)),before[:width]),
                      (f'[1:2:{size-2}]',odd,before),
                      (f'[{static_set}]',reordered,before),
                      (f'[{runtime_set}]',reordered,before),
                      ('[high:stride:low]',list(range(size-1,1,-2)),before))
        # Keep each operator in its own QCX program so large runtime sets do
        # not multiply their pairwise checks beyond the normal output budget.
        for symbol,function in OPERATIONS:
            statements,outputs,expected = [source,declarations],[],[]
            for selector,positions,original in selections:
                for expression,bits in nodes:
                    name = f'out{len(outputs)}'
                    statements.extend((f'bit[{len(original)}] {name}={registers.literal(original)};',
                                       f'{name}{selector}{symbol}{expression};'))
                    result = compound(original,positions,bits,function)
                    outputs.append((name,len(original)))
                    expected.append(sum(bit << i for i,bit in enumerate(result)))
            for i,(expression,value) in enumerate(preservation):
                name = f'preserved{i}'
                statements.append(f'bool {name}={expression};')
                outputs.append((name,None))
                expected.append(value)
            self.check_exact(' '.join(statements),outputs,expected)
            type(self).assignments += len(nodes)*len(selections)

    def test_overlap_self_references_and_original_destination_indices(self):
        cases = (('',(0,1,2,3)),('[3:-1:0]',(3,2,1,0)),('[{3,1,2,0}]',(3,1,2,0)),
                 ('[{high,1,2,low}]',(3,1,2,0)),('[high:-1:low]',(3,2,1,0)),
                 ('[low:2:high]',(0,2)))
        for pattern in range(16):
            before = [(pattern >> i)&1 for i in range(4)]
            for symbol,function in OPERATIONS:
                source,outputs,expected = ['int first=0; int last=3; uint high=3; int low=0;'],[],[]
                for selector,positions in cases:
                    for operation in ('~','rotl','^'):
                        name = f'out{len(outputs)}'
                        bounds = (0,1,3) if len(positions)==4 else (-1,4,3)
                        bits = registers.selected(pattern,4,bounds)
                        token = f'{name}[first:last]' if len(positions)==4 else f'{name}[-1:high+1:high]'
                        leaf = token,bits
                        if operation == '^':
                            mask = [i%2 for i in range(len(bits))]
                            node = registers.binary(leaf,(registers.literal(mask),mask),'^')
                        elif operation == 'rotl':
                            node = registers.unary(leaf,'rotl',1,f'int({name}=={registers.literal(before)})')
                        else:
                            node = registers.unary(leaf,'~')
                        source.extend((f'bit[4] {name}={registers.literal(before)};',f'{name}{selector}{symbol}{node[0]};'))
                        result = compound(before,positions,node[1],function)
                        outputs.append((name,4))
                        expected.append(sum(bit << i for i,bit in enumerate(result)))
                self.check_exact(' '.join(source),outputs,expected)
                type(self).assignments += len(outputs)
        for symbol,function in OPERATIONS:
            for pattern in range(16):
                before = [(pattern >> i)&1 for i in range(4)]
                positions = (before[0],before[1]+2)
                bits = [1-bit for bit in before[2:4]]
                result = compound(before,positions,bits,function)
                for selector in ('{int(out[0]),int(out[1])+2}',
                                 'int(out[0]):int(out[1])+2-int(out[0]):int(out[1])+2'):
                    self.check_exact(f'''bit[4] out={registers.literal(before)}; int first=2; int last=3;
                        out[{selector}]{symbol}~out[first:last];
                    ''',(('out',4),),(sum(bit << i for i,bit in enumerate(result)),))

    def test_loop_reentry_with_changing_counts_sources_and_transfers(self):
        source_pattern = 180
        bounds = ((0,1,3),(1,2,4),(-1,1,-1))
        positions = ((7,5,3,1),(6,3),(7,))
        for symbol,function in OPERATIONS:
            before = [(85 >> i)&1 for i in range(8)]
            values = []
            for distance,(selected,target) in enumerate(zip(bounds,positions)):
                bits = registers.unary(registers.unary(('',registers.selected(source_pattern,8,selected)),'rotl',distance),'~')[1]
                before = compound(before,target,bits,function)
                values.append(sum(bit << i for i,bit in enumerate(before)))
            self.check_exact('''bit[8] source="10110100"; bit[8] out="01010101";
                int first=0; int step=1; int last=3; int high=7; int stride=-2; int low=1; int visits=0;
                array[uint[8],3] values={0,0,0};
                while(visits<3) { out[high:stride:low]'''+symbol+'''~rotl(source[first:step:last],visits);
                    values[visits]=uint[8](out);
                    if(visits==0) { first=1; step=2; last=4; high=6; stride=-3; low=3; }
                    else { first=-1; step=1; last=-1; high=-1; stride=1; low=-1; }
                    visits+=1;
                }
                uint first_value=values[0]; uint second_value=values[1]; uint third_value=values[2];
                bool unchanged=source=="10110100";
            ''',tuple((name,None) for name in ('first_value','second_value','third_value','unchanged')),(*values,1))
            before = [(85 >> i)&1 for i in range(8)]
            source = [(11 >> i)&1 for i in range(4)]
            for i in (0,2):
                source[0] = 1-source[0]
                bits = [(bit & ((13 >> k)&1)) ^ ((3 >> k)&1) for k,bit in enumerate(source)]
                target = [position%8 for position in range(7-i,-i,-2)]
                before = compound(before,target,bits,function)
            self.check_exact('''bit[4] a="1011"; const bit[4] B="1101"; bit[8] out="01010101";
                int first=0; int last=3;
                for int i in [0:3] { if(i==1) { continue; } if(i==3) { break; }
                    a[0]=!a[0]; out[7-i:-2:1-i]'''+symbol+'''(a[first:last]&B)^"0011";
                }
            ''',(('out',8),('a',4)),(sum(bit << i for i,bit in enumerate(before)),11))

    def test_singleton_storage_native_steps_and_small_capacity_selections(self):
        c = self.module.QASM2QCXConverter
        for symbol,function in OPERATIONS:
            for target in ('one','one[0:0]','one[{-1}]','one[{index}]','one[index:index]'):
                for step in (str(c.QCX_INT_MIN),str(c.QCX_INT_MAX),'uint(-1)'):
                    self.check_exact('bit[1] one="1"; const bit[1] Z="0"; int index=-1; uint first=0; '
                                     f'{target}{symbol}~Z[first:{step}:first];',
                                     (('one',1),),(function(1,1),))
            for target in ('out[{index,0}]','out[index:-128:0]'):
                # A wide parent can use a short selection with a small buffer.
                with patch.object(c,'MAX_RUNTIME_BIT_RANGE_POSITIONS',2):
                    self.check_exact('bit[129] out="'+('0'*129)+'"; const bit[2] Z="00"; '
                                     f'int index=128; int first=0; int last=1; {target}{symbol}~Z[first:last];',
                                     (('out',129),),(function(0,1)*((1<<128)|1),))

    def test_nested_scalar_consumers_short_circuiting_and_unreached_domains(self):
        for symbol,function in OPERATIONS:
            before = [(85 >> i)&1 for i in range(8)]
            bits = registers.unary(('',[1,1,0,1]),'rotl',2)[1]
            result = compound(before,(7,5,3,1),bits,function)
            self.check_exact('''bit[4] a="1011"; const bit[4] B="1101"; bit[8] out="01010101";
                int first=0; int last=3; array[int,2] endpoints={7,1};
                out[endpoints[0]:-2:endpoints[1]]'''+symbol+'''rotl(a[first:last],
                    popcount(a[first:last]&B)+int(a[first:last]==B[first:last]));
            ''',(('out',8),('a',4)),(sum(bit << i for i,bit in enumerate(result)),11))
            self.check_exact('''bit[4] a="1011"; const bit[4] B="1101"; bit[8] out="01010101";
                int first=0; int last=3; int zero=0;
                if(false) { out[{first,-8,1/zero,999}]'''+symbol+'''rotl(a[first/zero:0:999],1/zero); }
                while(false) { out[first/zero:0:999]'''+symbol+'''a[first:0:last]<<-1; }
                for int i in [1:0] { out[7:-2:1]'''+symbol+'''~a[i/0:0:999]; }
                bool preserved=out=="01010101";
                out[first:first]'''+symbol+'''~a[int(false && a[first:0:last]==B):first];
            ''',(('preserved',None),('out',8)),(1,84 if symbol=='&=' else 85))

    def test_failure_order_and_suppression_of_combines_and_writes(self):
        cases = (
            ('out[{first,-8}]','rotl(a[first:last],1/zero)','assertion failed in ASSERT',()),
            ('out[{first,8}]','rotl(a[first:last],1/zero)','assertion failed in ASSERT',()),
            ('out[{first,big}]','rotl(a[first:last],1/zero)','UINT to INT conversion out of range',()),
            ('out[-1:8:last+4]','rotl(a[first:last],1/zero)','assertion failed in ASSERT',()),
            ('out[first:0:last]','rotl(a[first:last],1/zero)','assertion failed in ASSERT',()),
            ('out[last:1:first]','rotl(a[first:last],1/zero)','assertion failed in ASSERT',()),
            ('out[{first,7}]','rotl(a[first:last],1/zero)','integer division by zero in LET',(111,)),
            ('out','~a[first:last]','assertion failed in ASSERT',(111,222)),
            ('out[7:-2:1]','~a[first:first]','assertion failed in ASSERT',(111,222)),
            ('out[{7,5,3,1}]','~a[first:first]','assertion failed in ASSERT',(111,222)),
            ('out[{first,7}]','~a[first:last]','assertion failed in ASSERT',(111,222)),
            ('out[7:-2:last+2]','~a[first:last]','assertion failed in ASSERT',(111,222)),
            ('out[7:-2:1]','(a[first:first]&B)^rotl(a[first:last],1/zero)','assertion failed in ASSERT',(111,)),
            ('out[7:-2:1]','~a[first:0:last]','assertion failed in ASSERT',(111,)),
            ('out[7:-2:1]','a[first:last]>>-1','assertion failed in ASSERT',(111,)),
        )
        markers = {'ORIGINAL_DONE':111,'VALUE_DONE':222,'BITWISE':333,'WRITE':999}
        for symbol,_ in OPERATIONS:
            for target,rhs,error,expected in cases:
                with self.subTest(symbol=symbol,target=target,rhs=rhs):
                    lines = self.program('bit[4] a="1011"; const bit[4] B="1101"; bit[8] out="01010101"; '
                                         'int first=0; int last=3; int zero=0; uint big=uint(-1); '
                                         f'{target}{symbol}{rhs};')
                    modified = []
                    for line in lines:
                        modified.append(line)
                        for suffix,value in markers.items():
                            if line == '@QASM2QCX_REGISTER_ASSIGNMENT_0_'+suffix:
                                modified.append(f'PRINTLN {value}')
                    modified.append('PRINTLN 888')
                    result = self.execute(modified)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn(error,result.stderr)
                    self.assertNotIn('runtime error:',result.stderr)
                    self.assertEqual(result.stdout.split(),list(map(str,expected)))
                    type(self).checked += 1

    def test_later_iteration_duplicate_or_length_failure_has_no_partial_write(self):
        for symbol,_ in OPERATIONS:
            for update in ('low=3;','high=-1; stride=8; low=7;'):
                lines = self.program('''bit[4] a="1011"; bit[8] out="01010101"; int first=0; int last=3;
                    int high=7; int stride=-2; int low=1; int visits=0;
                    while(visits<2) { out[high:stride:low]'''+symbol+'~a[first:last]; '+update+' visits+=1; }')
                modified = []
                for line in lines:
                    if line.startswith('LET OUT7:') and ' := QASM2QCX_REGISTER_ASSIGNMENT_ORIGINAL_' in line:
                        modified.append('PRINTLN 999')
                    modified.append(line)
                    if line == '@QASM2QCX_REGISTER_ASSIGNMENT_0_END':
                        modified.append('PRINTLN 111')
                result = self.execute(modified)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('assertion failed in ASSERT',result.stderr)
                self.assertNotIn('runtime error:',result.stderr)
                self.assertEqual(result.stdout.split(),['999']*4+['111'])
                type(self).checked += 1

    def test_destination_and_rhs_capacity_failures_preserve_original_destination(self):
        for symbol,_ in OPERATIONS:
            for target,rhs,expected in (
                    ('out[last+4:-2:1]','rotl(a[first:last],1/zero)',()),
                    ('out[{first,7}]','~a[first:last]',(111,))):
                with self.subTest(symbol=symbol,target=target):
                    with patch.object(self.module.QASM2QCXConverter,'MAX_RUNTIME_BIT_RANGE_POSITIONS',2):
                        lines = self.program('bit[4] a="1011"; bit[8] out="01010101"; '
                                             'int first=0; int last=3; int zero=0; '+f'{target}{symbol}{rhs};')
                    modified = []
                    for line in lines:
                        modified.append(line)
                        if line == '@QASM2QCX_REGISTER_ASSIGNMENT_0_ORIGINAL_DONE':
                            modified.append('PRINTLN 111')
                        if line == '@QASM2QCX_REGISTER_ASSIGNMENT_0_WRITE':
                            modified.append('PRINTLN 999')
                    modified.append('PRINTLN 888')
                    result = self.execute(modified)
                    self.assertNotEqual(result.returncode,0)
                    self.assertIn('assertion failed in ASSERT',result.stderr)
                    self.assertNotIn('runtime error:',result.stderr)
                    self.assertEqual(result.stdout.split(),list(map(str,expected)))
                    type(self).checked += 1

    def test_measurement_feedback_and_fused_gate_consumers(self):
        self.check_exact('''include "stdgates.inc"; qubit[2] q; reset q; x q[0];
            bit[2] measured=measure q; bit[8] out="00000000"; int first=0; int last=1;
            int high=5; out[high:-1:high-1]|=rotl(measured[first:last],1);
            if(out[4]) { x q[1]; }
            bit[2] observed=measure q; bool feedback=observed=="11";
        ''',(('out',8),('feedback',None)),(16,1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = cmath.exp(1j*math.pi/4)
        for fused in (False,True):
            source = '''include "stdgates.inc"; qubit[2] q; h q;
                bit[4] a="1011"; bit[8] out="00000000"; int first=0; int last=3;
                out[last+4:-2:first+1]|=rotl(a[first:last],1);
                p(pi/4*int(out=="10101000")) q[0];
            '''
            BackendAuditTests.check(self,self.program(source,fused=fused),(0.5,phase/2,0.5,phase/2))


def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra',required=True,type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime expression bitwise compound programs checked: {BackendTests.checked}; destination registers checked: {BackendTests.assignments}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
