#!/usr/bin/env python3
"""Audit whole-register assignments from runtime-range expressions (#515).

Run with --bra /path/to/bra (a non-MPI build, including UBSan builds).
Independent little-endian bit vectors model register operations. Check both
ordinary assignments and declaration initializers, and instrument failures to
verify operand completion and suppression of destination writes.
"""

import argparse
import cmath
import itertools
import math
import pathlib
import random
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_runtime_range_equality_numerical as registers


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    assignments = 0
    setUp = registers.BackendTests.setUp
    execute = registers.BackendTests.execute
    program = registers.BackendTests.program
    check_exact = registers.BackendTests.check_exact

    def check_nodes(self, source, nodes, preservation=()):
        statements, outputs, expected = [source], [], []
        for index,(expression,bits) in enumerate(nodes):
            width = len(bits)
            initial, assigned = f'initial{index}', f'assigned{index}'
            statements.extend((f'bit[{width}] {initial}={expression};',
                               f'bit[{width}] {assigned}={registers.literal([0]*width)};',
                               f'{assigned}={expression};'))
            outputs.extend(((initial,width),(assigned,width)))
            pattern = sum(bit << i for i,bit in enumerate(bits))
            expected.extend((pattern,pattern))
        for index,(expression,value) in enumerate(preservation):
            name = f'preserved{index}'
            statements.append(f'bool {name}={expression};')
            outputs.append((name,None))
            expected.append(value)
        self.check_exact(' '.join(statements),outputs,expected)
        type(self).assignments += 2*len(nodes)

    def check_case(self, size, a, b, bounds, constants=(False,False)):
        reverse = (bounds[2],-bounds[1],bounds[0])
        left = ('a[first:step:last]',registers.selected(a,size,bounds))
        right = ('b[last:-step:first]',registers.selected(b,size,reverse))
        source = [('const ' if constants[0] else '')+f'bit[{size}] a="{a:0{size}b}";',
                  ('const ' if constants[1] else '')+f'bit[{size}] b="{b:0{size}b}";',
                  f'int first={bounds[0]}; int step={bounds[1]}; int last={bounds[2]};']
        fixed = ('MASK',left[1])
        source.append(f'const bit[{len(left[1])}] MASK={registers.literal(left[1])};')
        nodes = [left,registers.unary(left,'~'),registers.unary(right,'~'),
                 registers.unary(left,'<<',1),registers.unary(right,'>>',1),
                 registers.unary(left,'rotl',-1),registers.unary(right,'rotr',size+1)]
        for operation in ('&','|','^'):
            nodes.extend((registers.binary(left,right,operation),registers.binary(fixed,left,operation)))
        nodes.append(registers.binary(registers.binary(left,fixed,'&'),registers.unary(right,'~'),'^'))
        self.check_nodes(' '.join(source),nodes,
            ((f'a=="{a:0{size}b}" && b=="{b:0{size}b}"',1),
             (f'first=={bounds[0]} && step=={bounds[1]} && last=={bounds[2]}',1)))

    def test_exhaustive_small_patterns_and_source_kinds(self):
        for size in (1,2,3):
            for a,b in itertools.product(range(1 << size),repeat=2):
                self.check_case(size,a,b,(0,1,size-1),(bool(a%2),bool(b%2)))

    def test_native_widths_wide_ranges_negative_positions_and_repeated_sources(self):
        rng = random.Random(515)
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for size in (*range(1,native+2),65,129):
            bounds = (0,1,size-1) if size%2 else (size-1,-1,0)
            self.check_case(size,rng.getrandbits(size),rng.getrandbits(size),bounds,(bool(size%2),bool(size%3)))
        for size in (1,4,33,65,129):
            for bounds in ((-size,1,-1),(-1,-1,-size),(-size,1,size-1),
                           (size-1,-2,-size),(0,2,size-1),(size-1,-1,size-1)):
                self.check_case(size,rng.getrandbits(size),rng.getrandbits(size),bounds)

    def test_deeper_trees_and_intermediate_operations_at_native_endpoints(self):
        rng = random.Random(51501)
        c = self.module.QASM2QCXConverter
        for case in range(36):
            size = (1,2,4,33,65,129)[case%6]
            source = [f'int first=0; int last={size-1};']
            leaves, preservation = [], []
            for name in ('a','b','c','d'):
                pattern = rng.getrandbits(size)
                source.append(('const ' if rng.randrange(2) else '')+f'bit[{size}] {name}="{pattern:0{size}b}";')
                leaves.append((f'{name}[first:last]',registers.selected(pattern,size,(0,1,size-1))))
                preservation.append((f'{name}=="{pattern:0{size}b}"',1))
            def changed(node):
                for _ in range(3):
                    operation = rng.choice(('~','<<','>>','rotl','rotr'))
                    distance = rng.choice((0,1,size,size+1,c.QCX_UINT_MAX)) if operation in ('<<','>>') else rng.choice(
                        (0,-1,size+1,c.QCX_INT_MIN,c.QCX_INT_MAX))
                    token = 'uint(-1)' if distance == c.QCX_UINT_MAX else str(distance)
                    node = registers.unary(node,operation,distance,token)
                return node
            a,b,left_c,d = [changed(node) for node in leaves]
            balanced = changed(registers.binary(registers.binary(a,b,'&'),registers.binary(left_c,d,'^'),'|'))
            left = changed(registers.binary(registers.binary(registers.binary(a,b,'^'),left_c,'|'),d,'&'))
            right = changed(registers.binary(a,registers.binary(b,registers.binary(left_c,d,'|'),'&'),'^'))
            self.check_nodes(' '.join(source),(balanced,left,right),preservation)

    def test_self_assignment_nested_consumers_and_changing_ranges_in_loops(self):
        self.check_exact('''bit[4] a="1011"; const bit[4] B="1101"; int first=0; int last=3;
            a=rotl(a[first:last],int(a[first:last]!=B));
            bool self=a=="0111";
            bit[4] initialized=rotl(B[first:last],popcount(a[first:last]));
            bool nested=initialized=="1110";
            bit[4] out="0000"; bit[8] source="10110100";
            int step=1; int visits=0; array[uint[4],3] values={0,0,0};
            while(visits<3) {
                out=~rotl(source[first:step:last],visits);
                values[visits]=uint[4](out);
                if(visits==0) { first=1; step=2; last=7; }
                else { first=7; step=-2; last=1; }
                visits+=1;
            }
            uint first_value=values[0]; uint second_value=values[1]; uint third_value=values[2];
            first=0; last=3;
            for int i in [0:1] { a=~a[first:last]; }
            bool fresh=a=="0111";
        ''',tuple((name,None) for name in ('self','nested','first_value','second_value','third_value','fresh')),
            (1,1,11,6,3,1))
        self.check_exact('''bit[4] a="1011"; const bit[4] B="1101"; int first=0; int last=3;
            bit[4] out="0000"; int total=0;
            for int i in [0:2] { a[0]=!a[0]; out=(a[first:last]&B)^"0011"; total+=int(uint[4](out)); }
            bool fresh=out=="1011";
        ''',(('total',None),('fresh',None)),(32,1))

    def test_singletons_mixed_bound_types_and_known_width_cast_leaves(self):
        c = self.module.QASM2QCXConverter
        for step in (str(c.QCX_INT_MIN),str(c.QCX_INT_MAX),'uint(-1)'):
            self.check_exact(f'''bit[1] a="1"; int first=0; uint last=0;
                bit[1] out=~a[first:{step}:last]; out=~out[last:{step}:first];
            ''',(('out',1),),(1,))
        self.check_exact('''bit[4] a="1011"; int first=0; uint last=3; uint[4] mask=13;
            array[int,2] bounds={0,3}; angle[4] phase=13*pi/8;
            bit[4] out=a[bounds[first]:bounds[1]] & bit[4](mask);
            bool cast=out=="1001";
            out=a[first:last]^bit[4](phase);
            bool angle_cast=out=="0110";
            bit[8] repeated=a[-4:last];
            bool alias=repeated=="10111011";
        ''',(('cast',None),('angle_cast',None),('alias',None)),(1,1,1))

    def test_skipped_domains_and_empty_loops_preserve_current_destination(self):
        self.check_exact('''bit[4] a="1011"; const bit[4] B="1101";
            int first=0; int last=3; int zero=0; bit[4] out="1010";
            if(false) { out=rotl(a[first/zero:0:999],1/zero); }
            while(false) { out=(a[first:0:last]<<-1)^B; }
            for int i in [1:0] { out=~a[i/zero:0:999]; }
            bool preserved=out=="1010";
            out=~a[first+int(false && a[first:0:last]==B):last];
            bool bounds=out=="0100";
            out=rotl(a[first:last],int(true || a[first/zero:0:999]==B));
            bool distance=out=="0111";
            bool unchanged=a=="1011" && first==0 && last==3;
        ''',tuple((name,None) for name in ('preserved','bounds','distance','unchanged')),(1,1,1,1))

    def test_reached_errors_complete_rhs_before_final_size_check_and_writes(self):
        cases = (
            ('rotl(a[first:first],1/zero)',('VALUE_CHILD_0_SNAPSHOT_DONE',),'integer division by zero in LET',None),
            ('rotl(a[first:first],1)',('VALUE_CHILD_0_SNAPSHOT_DONE','VALUE_DONE'),'assertion failed in ASSERT',None),
            ('(a[first:first]&B)^rotl(a[first:last],1/zero)',
             ('VALUE_CHILD_0_CHILD_0_SNAPSHOT_DONE',),'assertion failed in ASSERT',None),
            ('a[first:0:last]<<1',(),'assertion failed in ASSERT',None),
            ('~a[first:-1:last]',(),'assertion failed in ASSERT',None),
            ('a[first:99]^B',(),'assertion failed in ASSERT',None),
            ('a[first:last]<<-1',('VALUE_CHILD_0_SNAPSHOT_DONE',),'assertion failed in ASSERT',None),
            ('~a[first:last]',(),'assertion failed in ASSERT',2),
        )
        labels = ('VALUE_CHILD_0_SNAPSHOT_DONE','VALUE_DONE','VALUE_CHILD_0_CHILD_0_SNAPSHOT_DONE')
        for expression,expected,error,capacity in cases:
            with self.subTest(expression=expression,capacity=capacity):
                program = self.module.openqasm3.parser.parse('OPENQASM 3; bit[4] a="1011"; const bit[4] B="1101"; '
                    'int first=0; int last=3; int zero=0; bit[4] out="1111"; out='+expression+';')
                converter = self.module.QASM2QCXConverter(program)
                if capacity is None:
                    converter.visit(program)
                else:
                    with patch.object(converter,'MAX_RUNTIME_BIT_RANGE_POSITIONS',capacity):
                        converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables,set())
                target = converter._QASM2QCXConverter__capitalize_variable_name('out')
                lines = []
                for line in converter:
                    if line.startswith(f'LET {target}:QASM2QCX_'):
                        lines.append('PRINTLN 999')
                    lines.append(line)
                    for index,label in enumerate(labels):
                        if line == '@QASM2QCX_REGISTER_ASSIGNMENT_0_'+label:
                            lines.append(f'PRINTLN {index+1}')
                lines.append('PRINTLN 888')
                result = self.execute(lines)
                self.assertNotEqual(result.returncode,0)
                self.assertIn(error,result.stderr)
                self.assertNotIn('runtime error:',result.stderr)
                self.assertEqual(result.stdout.split(),[str(labels.index(label)+1) for label in expected])
                type(self).checked += 1

    def test_failure_on_later_iteration_prevents_partial_commit(self):
        lines = self.program('''bit[4] a="1011"; const bit[4] B="1101"; int first=0; int last=3;
            bit[4] out="1111"; for int i in [0:1] { out=~(a[first:last]^B); last-=1; }
        ''')
        modified = []
        for line in lines:
            modified.append(line)
            if line == '@QASM2QCX_REGISTER_ASSIGNMENT_0_END':
                modified.append('PRINTLN 111')
        result = self.execute(modified)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('assertion failed in ASSERT',result.stderr)
        self.assertNotIn('runtime error:',result.stderr)
        self.assertEqual(result.stdout.split(),['111'])
        type(self).checked += 1

    def test_measurement_feedback_and_normal_fused_gate_consumers(self):
        self.check_exact('''include "stdgates.inc"; qubit[2] q; x q[0]; bit[2] measured=measure q;
            int first=0; int last=1;
            bit[2] initialized=rotl(measured[first:last],1);
            bit[2] out="00"; out=initialized[first:last]^"01";
            if(out=="11") { x q[1]; }
            bit[2] observed=measure q; bool feedback=observed=="11";
        ''',(('feedback',None),),(1,))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = cmath.exp(1j*math.pi/4)
        for fused in (False,True):
            source = '''include "stdgates.inc"; qubit[2] q; h q;
                bit[4] a="1011"; int first=0; int last=3;
                bit[4] initialized=~a[first:last]; bit[4] out="0000";
                out=rotl(initialized[first:last],1);
                p(pi/4*int(out=="1000")) q[0];
            '''
            BackendAuditTests.check(self,self.program(source,fused=fused),(0.5,phase/2,0.5,phase/2))


def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra',required=True,type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime-range expression assignment programs checked: {BackendTests.checked}; register results checked: {BackendTests.assignments}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
