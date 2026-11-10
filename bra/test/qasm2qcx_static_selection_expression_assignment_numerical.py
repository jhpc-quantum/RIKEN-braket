#!/usr/bin/env python3
"""Audit runtime-range expression assignments into static selections (#517).

Run with --bra /path/to/bra (a non-MPI build, including UBSan builds).
Independent bit-vector references model both RHS values and selected writes;
compare every destination bit, including those outside the selection.
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


def write_selected(before, positions, bits):
    assert len(positions) == len(bits) and len(set(positions)) == len(positions)
    result = before.copy()
    for position,bit in zip(positions,bits):
        result[position] = bit
    return result


def targets(width):
    size = 2*width+1
    reordered = list(range(1,size,2))
    reordered = reordered[1:]+reordered[:1]
    return ((f'0:{width-1}',list(range(width))),
            (f'-1:-2:{-(2*width-1)}',list(range(size-1,1,-2))),
            ('{'+','.join(str(i-size) for i in reordered)+'}',reordered))


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    assignments = 0
    setUp = registers.BackendTests.setUp
    execute = registers.BackendTests.execute
    program = registers.BackendTests.program
    check_exact = registers.BackendTests.check_exact

    def check_nodes(self, source, nodes, before, selections, preservation=()):
        statements, outputs, expected = [source], [], []
        for selector,positions in selections:
            for expression,bits in nodes:
                name = f'out{len(outputs)}'
                statements.extend((f'bit[{len(before)}] {name}={registers.literal(before)};',
                                   f'{name}[{selector}]={expression};'))
                result = write_selected(before,positions,bits)
                outputs.append((name,len(before)))
                expected.append(sum(bit << i for i,bit in enumerate(result)))
        for index,(expression,value) in enumerate(preservation):
            name = f'preserved{index}'
            statements.append(f'bool {name}={expression};')
            outputs.append((name,None))
            expected.append(value)
        self.check_exact(' '.join(statements),outputs,expected)
        type(self).assignments += len(nodes)*len(selections)

    def check_case(self, size, a, b, bounds, constants=(False,False)):
        reverse = (bounds[2],-bounds[1],bounds[0])
        left = ('a[first:step:last]',registers.selected(a,size,bounds))
        right = ('b[last:-step:first]',registers.selected(b,size,reverse))
        width = len(left[1])
        before = [(i%3 == 0)*1 for i in range(2*width+1)]
        source = [('const ' if constants[0] else '')+f'bit[{size}] a="{a:0{size}b}";',
                  ('const ' if constants[1] else '')+f'bit[{size}] b="{b:0{size}b}";',
                  f'int first={bounds[0]}; int step={bounds[1]}; int last={bounds[2]};',
                  f'const bit[{width}] MASK={registers.literal(left[1])};']
        fixed = ('MASK',left[1])
        nodes = [registers.unary(left,'~'),registers.unary(right,'rotl',-1),
                 registers.unary(left,'>>',width+1)]
        for operation in ('&','|','^'):
            nodes.append(registers.binary(left,right,operation))
        nodes.append(registers.binary(registers.binary(left,fixed,'&'),registers.unary(right,'~'),'^'))
        self.check_nodes(' '.join(source),nodes,before,targets(width),
            ((f'a=="{a:0{size}b}" && b=="{b:0{size}b}"',1),
             (f'first=={bounds[0]} && step=={bounds[1]} && last=={bounds[2]}',1)))

    def test_exhaustive_small_patterns_destinations_and_source_kinds(self):
        for size in (1,2,3):
            for a,b in itertools.product(range(1 << size),repeat=2):
                self.check_case(size,a,b,(0,1,size-1),(bool(a%2),bool(b%2)))

    def test_native_widths_wide_sources_negative_ranges_and_repeated_sources(self):
        rng = random.Random(517)
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for size in (*range(1,native+2),65,129):
            bounds = (0,1,size-1) if size%2 else (size-1,-1,0)
            self.check_case(size,rng.getrandbits(size),rng.getrandbits(size),bounds,(bool(size%2),bool(size%3)))
        for size in (1,4,33,65,129):
            for bounds in ((-size,1,-1),(-1,-1,-size),(-size,1,size-1),
                           (size-1,-2,-size),(0,2,size-1),(size-1,-1,size-1)):
                self.check_case(size,rng.getrandbits(size),rng.getrandbits(size),bounds)

    def test_deeper_trees_and_native_endpoint_distances(self):
        rng = random.Random(51701)
        c = self.module.QASM2QCXConverter
        for case in range(30):
            width = (1,2,4,33,65,129)[case%6]
            source = [f'int first=0; int last={width-1};']
            leaves, preservation = [], []
            for name in ('a','b','c','d'):
                pattern = rng.getrandbits(width)
                source.append(('const ' if rng.randrange(2) else '')+f'bit[{width}] {name}="{pattern:0{width}b}";')
                leaves.append((f'{name}[first:last]',registers.selected(pattern,width,(0,1,width-1))))
                preservation.append((f'{name}=="{pattern:0{width}b}"',1))
            def changed(node):
                for _ in range(3):
                    operation = rng.choice(('~','<<','>>','rotl','rotr'))
                    distance = rng.choice((0,1,width,width+1,c.QCX_UINT_MAX)) if operation in ('<<','>>') else rng.choice(
                        (0,-1,width+1,c.QCX_INT_MIN,c.QCX_INT_MAX))
                    token = 'uint(-1)' if distance == c.QCX_UINT_MAX else str(distance)
                    node = registers.unary(node,operation,distance,token)
                return node
            a,b,left_c,d = [changed(leaf) for leaf in leaves]
            nodes = (changed(registers.binary(registers.binary(a,b,'&'),registers.binary(left_c,d,'^'),'|')),
                     changed(registers.binary(a,registers.binary(b,registers.binary(left_c,d,'|'),'&'),'^')))
            before = [i%2 for i in range(2*width+1)]
            self.check_nodes(' '.join(source),nodes,before,targets(width),preservation)

    def test_overlapping_self_assignments_read_original_values(self):
        cases = ((0,1,2,((1,2,3),(3,0,1))),
                 (3,-1,0,((3,2,1,0),(2,0,3,1))),
                 (-1,-1,-4,((0,1,2,3),(1,3,0,2))),
                 (-1,4,3,((0,1),(3,0))))
        for pattern in range(16):
            before = [(pattern >> i)&1 for i in range(4)]
            source, outputs, expected = [], [], []
            for case,(first,step,last,selections) in enumerate(cases):
                source.append(f'int first{case}={first}; int step{case}={step}; int last{case}={last};')
                bits = registers.selected(pattern,4,(first,step,last))
                mask = [i%2 for i in range(len(bits))]
                for positions in selections:
                    for operation in ('~','rotl','^'):
                        name = f'out{len(outputs)}'
                        leaf = (f'{name}[first{case}:step{case}:last{case}]',bits)
                        if operation == 'rotl':
                            node = registers.unary(leaf,'rotl',1,f'int({name}=={registers.literal(before)})')
                        elif operation == '^':
                            node = registers.binary(leaf,(registers.literal(mask),mask),'^')
                        else:
                            node = registers.unary(leaf,'~')
                        selector = '{'+','.join(map(str,positions))+'}'
                        source.extend((f'bit[4] {name}={registers.literal(before)};',f'{name}[{selector}]={node[0]};'))
                        result = write_selected(before,positions,node[1])
                        outputs.append((name,4))
                        expected.append(sum(bit << i for i,bit in enumerate(result)))
            self.check_exact(' '.join(source),outputs,expected)
            type(self).assignments += len(outputs)

    def test_loop_reentry_changing_ranges_and_fresh_source_values(self):
        values, before = [], [(85 >> i)&1 for i in range(8)]
        for distance,bounds in enumerate(((0,1,3),(1,2,7),(7,-2,1))):
            node = registers.unary(registers.unary(('',registers.selected(180,8,bounds)),'rotl',distance),'~')
            before = write_selected(before,(7,1,5,0),node[1])
            values.append(sum(bit << i for i,bit in enumerate(before)))
        self.check_exact('''bit[8] source="10110100"; bit[8] out="01010101";
            int first=0; int step=1; int last=3; int visits=0;
            array[uint[8],3] values={0,0,0};
            while(visits<3) {
                out[{7,1,5,0}]=~rotl(source[first:step:last],visits);
                values[visits]=uint[8](out);
                if(visits==0) { first=1; step=2; last=7; }
                else { first=7; step=-2; last=1; }
                visits+=1;
            }
            uint first_value=values[0]; uint second_value=values[1]; uint third_value=values[2];
            bool unchanged=source=="10110100";
        ''',tuple((name,None) for name in ('first_value','second_value','third_value','unchanged')),(*values,1))
        source = [(11 >> i)&1 for i in range(4)]
        before = [(240 >> i)&1 for i in range(8)]
        total = 0
        for _ in range(3):
            source[0] = 1-source[0]
            bits = [(bit & ((13 >> i)&1)) ^ ((3 >> i)&1) for i,bit in enumerate(source)]
            before = write_selected(before,(7,5,3,1),bits)
            total += sum(bit << i for i,bit in enumerate(before))
        self.check_exact('''bit[4] a="1011"; bit[8] out="11110000"; const bit[4] B="1101";
            int first=0; int last=3; int total=0;
            for int i in [0:2] {
                a[0]=!a[0]; out[7:-2:1]=(a[first:last]&B)^"0011";
                total+=int(uint[8](out));
            }
        '''+f'bool fresh=out=={registers.literal(before)};',(('total',None),('fresh',None)),(total,1))

    def test_singletons_mixed_domains_and_skipped_execution(self):
        c = self.module.QASM2QCXConverter
        for step in (str(c.QCX_INT_MIN),str(c.QCX_INT_MAX),'uint(-1)'):
            self.check_exact(f'''bit[1] one="1"; int first=0; uint last=0;
                one[{{0}}]=~one[first:{step}:last];
                bit[4] out="1010"; out[0:0]=~one[last:{step}:first];
            ''',(('one',1),('out',4)),(0,11))
        self.check_exact('''bit[4] a="1011"; const bit[4] B="1101";
            bit[8] out="01010101"; int first=0; int last=3; int zero=0;
            if(false) { out[7:-2:1]=rotl(a[first/zero:0:999],1/zero); }
            while(false) { out[{7,1,5,0}]=a[first:0:last]<<-1; }
            for int i in [1:0] { out[0:3]=~a[i/zero:0:999]; }
            bool preserved=out=="01010101";
            out[3:-1:0]=rotl(a[first:last],int(true || a[first:0:last]==B));
            bool reversed=out=="01011110";
        ''',(('preserved',None),('reversed',None)),(1,1))

    def test_reached_errors_and_final_length_checks_precede_all_selected_writes(self):
        cases = (
            ('out[{7,1,5,0}]','rotl(a[first:first],1/zero)',('VALUE_CHILD_0_SNAPSHOT_DONE',),'integer division by zero in LET',None),
            ('out[{7,1,5,0}]','rotl(a[first:first],1)',('VALUE_CHILD_0_SNAPSHOT_DONE','VALUE_DONE'),'assertion failed in ASSERT',None),
            ('out[{7,1,5,0}]','(a[first:first]&B)^rotl(a[first:last],1/zero)',
             ('VALUE_CHILD_0_CHILD_0_SNAPSHOT_DONE',),'assertion failed in ASSERT',None),
            ('out[7:-2:1]','~a[first:0:last]',(),'assertion failed in ASSERT',None),
            ('out[7:-2:1]','a[first:last]>>-1',('VALUE_CHILD_0_SNAPSHOT_DONE',),'assertion failed in ASSERT',None),
            ('out[7:-2:1]','a[first:99]^B',(),'assertion failed in ASSERT',None),
            ('out[7:7]','~a[first:last]',(),'assertion failed in ASSERT',2),
        )
        labels = ('VALUE_CHILD_0_SNAPSHOT_DONE','VALUE_DONE','VALUE_CHILD_0_CHILD_0_SNAPSHOT_DONE')
        for target,expression,expected,error,capacity in cases:
            with self.subTest(target=target,expression=expression,capacity=capacity):
                program = self.module.openqasm3.parser.parse('OPENQASM 3; bit[4] a="1011"; const bit[4] B="1101"; '
                    'bit[8] out="01010101"; int first=0; int last=3; int zero=0; '+target+'='+expression+';')
                converter = self.module.QASM2QCXConverter(program)
                if capacity is None:
                    converter.visit(program)
                else:
                    with patch.object(converter,'MAX_RUNTIME_BIT_RANGE_POSITIONS',capacity):
                        converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables,set())
                name = converter._QASM2QCXConverter__capitalize_variable_name('out')
                lines, active = [], False
                for line in converter:
                    active = active or 'QASM2QCX_REGISTER_ASSIGNMENT_0' in line
                    if active and line.startswith(f'LET {name}:'):
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

    def test_later_iteration_failure_prevents_partial_commit(self):
        lines = self.program('''bit[4] a="1011"; bit[8] out="01010101"; int first=0; int last=3;
            int visits=0; while(visits<2) { out[{7,1,5,0}]=~a[first:last]; last-=1; visits+=1; }
        ''')
        modified = []
        for line in lines:
            if line.startswith('LET OUT7:') and ' := QASM2QCX_' in line:
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

    def test_measurement_feedback_and_normal_fused_gate_consumers(self):
        self.check_exact('''include "stdgates.inc"; qubit[2] q; reset q; x q[0];
            bit[2] measured=measure q; bit[8] out="00000000"; int first=0; int last=1;
            out[5:-1:4]=rotl(measured[first:last],1);
            if(out[4]) { x q[1]; }
            bit[2] observed=measure q; bool feedback=observed=="11";
        ''',(('out',8),('feedback',None)),(16,1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = cmath.exp(1j*math.pi/4)
        for fused in (False,True):
            source = '''include "stdgates.inc"; qubit[2] q; h q;
                bit[4] a="1011"; bit[8] out="00000000"; int first=0; int last=3;
                out[7:-2:1]=rotl(a[first:last],1);
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
    print(f'static-selection expression assignment programs checked: {BackendTests.checked}; destination registers checked: {BackendTests.assignments}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
