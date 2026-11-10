#!/usr/bin/env python3
"""Audit runtime-range expression assignments into runtime index sets (#519).

Run with --bra /path/to/bra (a non-MPI build, including UBSan builds).
Python bit-vector references check every destination bit and source preservation.
Instrumented failure cases check evaluation order and the absence of writes.
"""

import argparse
import cmath
import itertools
import math
import pathlib
import random
import resource
import unittest

import qasm2qcx_runtime_range_equality_numerical as registers
from qasm2qcx_static_selection_expression_assignment_numerical import write_selected


def runtime_targets(width):
    size = 2*width+1
    positions = list(range(1,size,2))
    positions = positions[1:]+positions[:1]
    indices = [position-size if i%2 == 0 else position for i,position in enumerate(positions)]
    declarations = (f'int destination={indices[0]}; uint positive={positions[-1]}; '
                    f'array[int,{width}] addresses={{'+','.join(map(str,indices))+'};')
    mixed = ['destination'] + [str(index) for index in indices[1:]]
    if width > 1:
        mixed[-1] = 'positive'
    return declarations, (('{'+','.join(mixed)+'}',positions),
                          ('{'+','.join(f'addresses[{i}]' for i in range(width))+'}',positions))


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    assignments = 0
    setUp = registers.BackendTests.setUp
    execute = registers.BackendTests.execute
    program = registers.BackendTests.program
    check_exact = registers.BackendTests.check_exact

    def test_documented_runtime_set_expression_assignment_example(self):
        document = (pathlib.Path(__file__).resolve().parents[1] / 'qcx/qasm2qcx.md').read_text()
        marker = '<!-- runtime-set-expression-assignment-example -->'
        source = document.split(marker,1)[1].split('```qasm\n',1)[1].split('```',1)[0]
        source = source.replace('OPENQASM 3.0;','',1)
        self.check_exact(source,(('result',8),('a',4),('one',1),('captured',4),
                                 ('rotated',None),('combined',None),('untouched',None)),(87,2,1,3,1,1,1))

    def check_nodes(self, source, nodes, before, preservation=()):
        declarations,selections = runtime_targets(len(nodes[0][1]))
        statements,outputs,expected = [source,declarations],[],[]
        for selector,positions in selections:
            for expression,bits in nodes:
                name = f'out{len(outputs)}'
                statements.extend((f'bit[{len(before)}] {name}={registers.literal(before)};',
                                   f'{name}[{selector}]={expression};'))
                result = write_selected(before,positions,bits)
                outputs.append((name,len(before)))
                expected.append(sum(bit << i for i,bit in enumerate(result)))
        for i,(expression,value) in enumerate(preservation):
            name = f'preserved{i}'
            statements.append(f'bool {name}={expression};')
            outputs.append((name,None))
            expected.append(value)
        self.check_exact(' '.join(statements),outputs,expected)
        type(self).assignments += len(nodes)*len(selections)

    def check_case(self, size, a, b, bounds, constants=(False,False)):
        reverse = bounds[2],-bounds[1],bounds[0]
        left = 'a[first:step:last]',registers.selected(a,size,bounds)
        right = 'b[last:-step:first]',registers.selected(b,size,reverse)
        width = len(left[1])
        source = [('const ' if constants[0] else '')+f'bit[{size}] a="{a:0{size}b}";',
                  ('const ' if constants[1] else '')+f'bit[{size}] b="{b:0{size}b}";',
                  f'int first={bounds[0]}; int step={bounds[1]}; int last={bounds[2]};',
                  f'const bit[{width}] MASK={registers.literal(left[1])};']
        nodes = [registers.unary(left,'~'),registers.unary(right,'rotl',-1),
                 registers.unary(left,'>>',width+1)]
        nodes.extend(registers.binary(left,right,operation) for operation in ('&','|','^'))
        nodes.append(registers.binary(registers.binary(left,('MASK',left[1]),'&'),
                                      registers.unary(right,'~'),'^'))
        self.check_nodes(' '.join(source),nodes,[int(i%3 == 0) for i in range(2*width+1)],
            ((f'a=="{a:0{size}b}" && b=="{b:0{size}b}"',1),
             (f'first=={bounds[0]} && step=={bounds[1]} && last=={bounds[2]}',1)))

    def test_exhaustive_small_patterns_and_source_kinds(self):
        for size in (1,2,3):
            for a,b in itertools.product(range(1 << size),repeat=2):
                self.check_case(size,a,b,(0,1,size-1),(bool(a%2),bool(b%2)))

    def test_native_widths_wide_sources_and_negative_ranges(self):
        rng = random.Random(519)
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for size in (*range(1,native+2),65,129):
            bounds = (0,1,size-1) if size%2 else (size-1,-1,0)
            self.check_case(size,rng.getrandbits(size),rng.getrandbits(size),bounds,(bool(size%2),bool(size%3)))
        for size in (1,4,33,65,129):
            for bounds in ((-size,1,-1),(-1,-1,-size),(-size,1,size-1),
                           (size-1,-2,-size),(0,2,size-1),(size-1,-1,size-1)):
                self.check_case(size,rng.getrandbits(size),rng.getrandbits(size),bounds)

    def test_deep_trees_and_native_endpoint_distances(self):
        rng = random.Random(51901)
        c = self.module.QASM2QCXConverter
        for case in range(18):
            width = (1,2,4,33,65,129)[case%6]
            source = [f'int first=0; int last={width-1};']
            leaves,preservation = [],[]
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
            self.check_nodes(' '.join(source),nodes,[i%2 for i in range(2*width+1)],preservation)

    def test_overlap_and_indices_read_from_original_destination(self):
        for pattern in range(16):
            before = [(pattern >> i)&1 for i in range(4)]
            statements,outputs,expected = ['int first=0; int last=3; int index=-1;'],[],[]
            for positions in ((3,1,2,0),(3,0,1,2)):
                for operation in ('~','rotl','^'):
                    name = f'out{len(outputs)}'
                    leaf = f'{name}[first:last]',before
                    if operation == '^':
                        node = registers.binary(leaf,('"1010"',[0,1,0,1]),'^')
                    elif operation == 'rotl':
                        node = registers.unary(leaf,'rotl',1,f'int({name}=={registers.literal(before)})')
                    else:
                        node = registers.unary(leaf,'~')
                    selector = '{index,'+','.join(map(str,positions[1:]))+'}'
                    statements.extend((f'bit[4] {name}={registers.literal(before)};',f'{name}[{selector}]={node[0]};'))
                    result = write_selected(before,positions,node[1])
                    outputs.append((name,4))
                    expected.append(sum(bit << i for i,bit in enumerate(result)))
            self.check_exact(' '.join(statements),outputs,expected)
            type(self).assignments += len(outputs)
        self.check_exact('''bit[4] out="0000"; const bit[2] C="00"; int first=0; int last=1;
            out[{0,int(out[0])+1}]=~C[first:last];
        ''',(('out',4),),(3,))

    def test_loop_reentry_refreshes_indices_ranges_and_source_values(self):
        source = [(11 >> i)&1 for i in range(4)]
        before = [(85 >> i)&1 for i in range(8)]
        total = 0
        for iteration in range(3):
            source[0] = 1-source[0]
            bits = [bit ^ ((3 >> i)&1) for i,bit in enumerate(source)]
            before = write_selected(before,(7-iteration,iteration,4,3),bits)
            total += sum(bit << i for i,bit in enumerate(before))
        self.check_exact('''bit[4] a="1011"; bit[8] out="01010101";
            int first=0; int last=3; int high=7; int low=0; int total=0;
            for int i in [0:2] {
                a[0]=!a[0]; out[{high,low,4,3}]=a[first:last]^"0011";
                total+=int(uint[8](out)); high-=1; low+=1;
            }
        '''+f'bool fresh=out=={registers.literal(before)};',(('total',None),('fresh',None)),(total,1))
        values,before = [],[(85 >> i)&1 for i in range(8)]
        for distance,bounds in enumerate(((0,1,3),(1,2,7),(7,-2,1))):
            bits = registers.unary(registers.unary(('',registers.selected(180,8,bounds)),'rotl',distance),'~')[1]
            before = write_selected(before,(7-distance,distance,4,3),bits)
            values.append(sum(bit << i for i,bit in enumerate(before)))
        self.check_exact('''bit[8] source="10110100"; bit[8] out="01010101";
            int first=0; int step=1; int last=3; int visits=0; uint high=7;
            array[uint[8],3] values={0,0,0};
            while(visits<3) {
                out[{high,visits,4,3}]=~rotl(source[first:step:last],visits);
                values[visits]=uint[8](out); high-=1;
                if(visits==0) { first=1; step=2; last=7; }
                else { first=7; step=-2; last=1; }
                visits+=1;
            }
            uint first_value=values[0]; uint second_value=values[1]; uint third_value=values[2];
            bool unchanged=source=="10110100";
        ''',tuple((name,None) for name in ('first_value','second_value','third_value','unchanged')),(*values,1))

    def test_nested_scalar_counts_retain_captured_destination_addresses(self):
        before = [(85 >> i)&1 for i in range(8)]
        bits = registers.unary(('',[1,1,0,1]),'rotl',2)[1]
        result = write_selected(before,(7,1,5,0),bits)
        self.check_exact('''bit[4] a="1011"; const bit[4] B="1101"; bit[8] out="01010101";
            int first=0; int last=3; array[int,2] addresses={7,5};
            out[{addresses[first],int(out[0]),addresses[1],first}]=rotl(a[first:last],
                popcount(a[first:last]&B)+int(a[first:last]==B[first:last]));
        ''',(('out',8),('a',4)),(sum(bit << i for i,bit in enumerate(result)),11))

    def test_singletons_short_circuiting_and_skipped_execution(self):
        c = self.module.QASM2QCXConverter
        for step in (str(c.QCX_INT_MIN),str(c.QCX_INT_MAX),'uint(-1)'):
            self.check_exact(f'''bit[1] one="1"; int first=0; uint last=0; int destination=-1;
                one[{{destination}}]=~one[first:{step}:last];
                bit[4] out="1010"; out[{{last}}]=~one[last:{step}:first];
            ''',(('one',1),('out',4)),(0,11))
        self.check_exact('''bit[4] a="1011"; const bit[4] B="1101";
            bit[8] out="01010101"; int first=0; int last=3; int zero=0;
            if(false) { out[{first,1/zero,0,999}]=rotl(a[first/zero:0:999],1/zero); }
            while(false) { out[{first,-8,0,999}]=a[first:0:last]<<-1; }
            for int i in [1:0] { out[{i,1/0,-8,0}]=~a[i/0:0:999]; }
            bool preserved=out=="01010101";
            out[{last,2,1,first}]=rotl(a[first:last],int(true || a[first:0:last]==B));
            bool reversed=out=="01011110";
        ''',(('preserved',None),('reversed',None)),(1,1))

    def test_destination_validation_precedes_rhs_and_no_failure_writes(self):
        cases = (
            ('first,8','rotl(a[first:last],1/zero)','assertion failed in ASSERT',False),
            ('first,-9','rotl(a[first:last],1/zero)','assertion failed in ASSERT',False),
            ('first,-8','rotl(a[first:last],1/zero)','assertion failed in ASSERT',False),
            ('first,big','rotl(a[first:last],1/zero)','UINT to INT conversion out of range',False),
            ('first,addresses[last]','rotl(a[first:last],1/zero)','assertion failed in ASSERT',False),
            ('first,1/zero','~a[first:last]','integer division by zero in LET',False),
            # All destination expressions are evaluated before duplicate checks.
            ('first,-8,1/zero','~a[first:last]','integer division by zero in LET',False),
            ('first,1','rotl(a[first:first],1/zero)','integer division by zero in LET',False),
            ('first,1','~a[first:first]','assertion failed in ASSERT',True),
            ('first,1','(a[first:first]&B)^rotl(a[first:last],1/zero)','assertion failed in ASSERT',False),
            ('first,1','~a[first:0:last]','assertion failed in ASSERT',False),
            ('first,1','a[first:last]>>-1','assertion failed in ASSERT',False),
            ('first,1','a[first:99]^B','assertion failed in ASSERT',False),
        )
        for entries,rhs,error,complete in cases:
            with self.subTest(entries=entries,rhs=rhs):
                source = ('bit[4] a="1011"; const bit[4] B="1101"; bit[8] out="01010101"; '
                          'int first=0; int last=3; int zero=0; uint big=uint(-1); '
                          'array[int,2] addresses={0,1}; '+f'out[{{{entries}}}]={rhs};')
                lines = self.program(source)
                modified = []
                for line in lines:
                    modified.append(line)
                    if line == '@QASM2QCX_REGISTER_ASSIGNMENT_0_VALUE_DONE':
                        modified.append('PRINTLN 111')
                    if line == '@QASM2QCX_REGISTER_ASSIGNMENT_0_WRITE':
                        modified.append('PRINTLN 999')
                modified.append('PRINTLN 888')
                result = self.execute(modified)
                self.assertNotEqual(result.returncode,0)
                self.assertIn(error,result.stderr)
                self.assertNotIn('runtime error:',result.stderr)
                self.assertEqual(result.stdout.split(),['111'] if complete else [])
                type(self).checked += 1

    def test_later_iteration_duplicate_prevents_partial_writes(self):
        lines = self.program('''bit[4] a="1011"; bit[8] out="01010101";
            int first=0; int last=3; int index=7; int visits=0;
            while(visits<2) { out[{index,1,5,0}]=~a[first:last]; index=1; visits+=1; }
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

    def test_measurement_feedback_and_fused_gate_consumers(self):
        self.check_exact('''include "stdgates.inc"; qubit[2] q; reset q; x q[0];
            bit[2] measured=measure q; bit[8] out="00000000"; int first=0; int last=1;
            int high=5; out[{high,high-1}]=rotl(measured[first:last],1);
            if(out[4]) { x q[1]; }
            bit[2] observed=measure q; bool feedback=observed=="11";
        ''',(('out',8),('feedback',None)),(16,1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = cmath.exp(1j*math.pi/4)
        for fused in (False,True):
            source = '''include "stdgates.inc"; qubit[2] q; h q;
                bit[4] a="1011"; bit[8] out="00000000"; int first=0; int last=3;
                out[{last+4,5,3,1}]=rotl(a[first:last],1);
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
    print(f'runtime-set expression assignment programs checked: {BackendTests.checked}; destination registers checked: {BackendTests.assignments}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
