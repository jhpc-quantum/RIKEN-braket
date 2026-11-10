#!/usr/bin/env python3
"""Audit runtime-range register equality and inequality (#513).

Run with --bra /path/to/bra (a non-MPI build, including UBSan builds).
References use independent little-endian bit lists, not converter folding,
UINT packing, or the generated QCX. Both complete operands must finish before
length validation and before a mismatching bit can end a comparison.
"""

import argparse
import cmath
import itertools
import math
import operator
import pathlib
import random
import resource
import unittest

import qasm2qcx_angle_bit_cast_numerical as backend


def literal(bits):
    return '"' + ''.join(str(bit) for bit in reversed(bits)) + '"'


def selected(pattern, size, bounds):
    first, step, last = bounds
    indices = list(range(first, last + (1 if step > 0 else -1), step))
    assert indices and all(-size <= index < size for index in indices)
    return [(pattern >> (index % size)) & 1 for index in indices]


def unary(node, operation, distance=0, token=None):
    expression, bits = node
    token = str(distance) if token is None else token
    if operation == '~':
        return f'~({expression})', [1 - bit for bit in bits]
    if operation == '<<':
        assert distance >= 0
        result = [bits[i - distance] if i >= distance else 0 for i in range(len(bits))]
    elif operation == '>>':
        assert distance >= 0
        result = [bits[i + distance] if i + distance < len(bits) else 0 for i in range(len(bits))]
    else:
        assert operation in ('rotl', 'rotr')
        offset = distance if operation == 'rotl' else -distance
        return f'{operation}({expression},{token})', [bits[(i-offset) % len(bits)] for i in range(len(bits))]
    return f'({expression}) {operation} ({token})', result


def binary(left, right, operation):
    assert len(left[1]) == len(right[1])
    function = {'&': operator.and_, '|': operator.or_, '^': operator.xor}[operation]
    return f'({left[0]}) {operation} ({right[0]})', [function(a,b) for a,b in zip(left[1],right[1])]


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    comparisons = 0
    setUp = backend.BackendTests.setUp
    execute = backend.BackendTests.execute
    program = backend.BackendTests.program
    check_exact = backend.BackendTests.check_exact

    def check_pairs(self, source, pairs):
        statements, outputs, expected = [source], [], []
        for left, right in pairs:
            self.assertEqual(len(left[1]), len(right[1]))
            for operation in ('==', '!='):
                name = f'out{len(outputs)}'
                statements.append(f'bool {name}=({left[0]}) {operation} ({right[0]});')
                outputs.append((name,None))
                equal = left[1] == right[1]
                expected.append(int(equal if operation == '==' else not equal))
        statements.append('bool preserved=' + ' && '.join(
            f'{name}=={pattern}' for name,pattern in self.preserved) + ';')
        outputs.append(('preserved',None))
        expected.append(1)
        self.check_exact(' '.join(statements),outputs,expected)
        type(self).comparisons += len(expected) - 1

    def check_case(self, size, a, b, bounds, *, constants=(False,False)):
        reverse = (bounds[2], -bounds[1], bounds[0])
        left_bits, right_bits = selected(a,size,bounds), selected(b,size,reverse)
        source = [('const ' if constants[0] else '') + f'bit[{size}] a="{a:0{size}b}";',
                  ('const ' if constants[1] else '') + f'bit[{size}] b="{b:0{size}b}";',
                  f'int first={bounds[0]}; int step={bounds[1]}; int last={bounds[2]};',
                  f'const bit[{len(left_bits)}] MASK={literal(left_bits)};']
        self.preserved = (('a',f'"{a:0{size}b}"'), ('b',f'"{b:0{size}b}"'))
        left = ('a[first:step:last]',left_bits)
        right = ('b[last:-step:first]',right_bits)
        fixed = ('MASK',left_bits)
        pairs = [(left,right),(right,left),(left,fixed),(fixed,left),
                 (left,(literal(right_bits),right_bits)),
                 (unary(left,'~'),unary(right,'~')),
                 (unary(left,'<<',1),unary(right,'>>',1)),
                 (unary(left,'rotl',-1),unary(right,'rotr',size+1)),
                 (binary(left,fixed,'^'),binary(right,fixed,'|'))]
        self.check_pairs(' '.join(source),pairs)

    def test_exhaustive_small_patterns_and_source_kinds(self):
        for width in (1,2,3):
            for a,b in itertools.product(range(1 << width),repeat=2):
                self.check_case(width,a,b,(0,1,width-1),constants=(bool(a%2),bool(b%2)))

    def test_every_native_width_wide_ranges_and_repeated_source_indices(self):
        rng = random.Random(513)
        native = self.module.QASM2QCXConverter.QCX_UINT_WIDTH
        for size in (*range(1,native+2),65,129):
            bounds = (0,1,size-1) if size%2 else (size-1,-1,0)
            self.check_case(size,rng.getrandbits(size),rng.getrandbits(size),bounds,
                            constants=(bool(size%2),bool(size%3)))
        for size in (1,4,33,65,129):
            for bounds in ((-size,1,-1),(-1,-1,-size),(-size,1,size-1),
                           (size-1,-2,-size),(0,2,size-1),(size-1,-1,size-1)):
                self.check_case(size,rng.getrandbits(size),rng.getrandbits(size),bounds)

    def test_nested_trees_and_arbitrary_intermediate_operations(self):
        rng = random.Random(51301)
        c = self.module.QASM2QCXConverter
        for case in range(36):
            size = (1,2,4,33,65,129)[case%6]
            source = [f'int first=0; int last={size-1};']
            self.preserved, leaves = [], []
            for name in ('a','b','c','d'):
                pattern = rng.getrandbits(size)
                source.append(('const ' if rng.randrange(2) else '') + f'bit[{size}] {name}="{pattern:0{size}b}";')
                self.preserved.append((name,f'"{pattern:0{size}b}"'))
                leaves.append((f'{name}[first:last]',selected(pattern,size,(0,1,size-1))))
            def changed(node):
                for _ in range(3):
                    op = rng.choice(('~','<<','>>','rotl','rotr'))
                    distance = rng.choice((0,1,size,size+1,c.QCX_UINT_MAX)) if op in ('<<','>>') else rng.choice(
                        (0,1,-1,size+1,c.QCX_INT_MIN,c.QCX_INT_MAX))
                    token = 'uint(-1)' if distance == c.QCX_UINT_MAX else str(distance)
                    node = unary(node,op,distance,token)
                return node
            a,b,c_leaf,d = [changed(leaf) for leaf in leaves]
            left = changed(binary(binary(a,b,'&'),binary(c_leaf,d,'^'),'|'))
            right = changed(binary(a,binary(c_leaf,d,'|'),'^'))
            # A matching literal gives a true result for every generated tree,
            # even when two independent random expressions rarely agree.
            self.check_pairs(' '.join(source),[(left,right),(right,left),
                (left,(literal(left[1]),left[1])),((literal(right[1]),right[1]),right)])

    def test_nested_consumers_self_assignment_indexed_writes_and_reentry(self):
        self.check_exact('''bit[4] a="1011"; const bit[4] B="1011"; int first=0; int last=3;
            bool same=false; same=rotl(a[first:last],int(same))==B;
            same=rotl(a[first:last],int(same))!=B;
            bit flag=a[first:last]==B; int numeric=int(flag);
            array[bool,2] stored={false,false}; stored[int(flag)]=a[first:last]==B;
            bool held=stored[1];
            uint count=popcount(a[first:last] << int(a[first:last]==B));
            bool nested=rotl(a[first:last],int(a[first:last]==B))=="0111";
            bool fixed=(a << int(a[first:last]==B))=="0110";
            int total=0; while(first<=last) {
                if(a[first:last]==B[first:last]) { total+=1; }
                first+=1;
            }
            first=0; for int i in [0:1] {
                for bit v in a[first:last] { total+=int(v); }
                a[0]=!a[0];
                if(a[first:last]!=B[first:last]) { total+=10; }
            }
            bool fresh=a[first:last]==B;
        ''',tuple((name,None) for name in ('same','flag','numeric','held','count','nested','fixed','total','fresh')),
            (1,1,1,1,2,1,1,19,1))

    def test_overlapping_sources_and_destination_addresses(self):
        self.preserved = (('a','"10110100"'),)
        source = 'bit[8] a="10110100"; int first=0; int last=7; int begin=-8;'
        direct = ('a[first:last]',selected(180,8,(0,1,7)))
        reverse = ('a[last:-1:first]',selected(180,8,(7,-1,0)))
        repeated = ('a[begin:last]',selected(180,8,(-8,1,7)))
        repeated_reverse = ('a[last:-1:begin]',selected(180,8,(7,-1,-8)))
        self.check_pairs(source,[(direct,direct),(direct,reverse),
            (unary(direct,'rotl',3),unary(reverse,'rotr',2)),
            (repeated,repeated_reverse),(repeated,repeated)])
        self.check_exact('''bit[4] a="1011"; int first=0; int last=3;
            a[int(a[first:last]=="1011")]=a[first:last]=="0000";
            uint changed=uint[4](a);
            array[bool,2] stored={false,false};
            stored[int(a[first:last]=="1001")]=rotl(a[first:last],int(a[first:last]=="1001"))=="0011";
            bool held=stored[1];
            first=int(a[first:last]=="1001");
            bool fresh=a[first:last]=="100";
        ''',(('changed',None),('held',None),('first',None),('fresh',None)),(9,1,1,1))

    def test_native_endpoint_steps_mixed_domains_and_known_width_leaves(self):
        c = self.module.QASM2QCXConverter
        for step in (str(c.QCX_INT_MIN),str(c.QCX_INT_MAX),'uint(-1)'):
            self.check_exact(f'''bit[1] a="1"; const bit[1] B="1"; int first=0; uint last=0;
                bool same=a[first:{step}:last]==B;
                bool different="0"!=a[last:{step}:first];
            ''',(('same',None),('different',None)),(1,1))
        self.check_exact('''bit[4] a="1011"; int first=0; uint last=3;
            array[int,2] bounds={0,3}; uint[4] u=11; angle[4] angle_value=11*pi/8;
            bool cast=a[first:last]==bit[4](u);
            bool angle_cast=a[first:last]==bit[4](angle_value);
            bool indexed=a[bounds[first]:bounds[1]]==a[{0,1,2,3}];
            bool repeated=a[-4:last]==a[{-4,-3,-2,-1,0,1,2,3}];
        ''',tuple((name,None) for name in ('cast','angle_cast','indexed','repeated')),(1,1,1,1))

    def test_skipped_domains_and_empty_loops_preserve_destinations(self):
        self.check_exact('''bit[4] a="1011"; int first=0; int last=3; int zero=0;
            array[int,1] bounds={0}; int bad=99;
            bool x=false && ((a[first/zero:0:999] << -1)=="1");
            bool y=true || (rotl(a[first:last],bounds[bad])!=a[first:last]);
            bit z=false && (a[first:last]==(a << bounds[bad]));
            bool preserved=true;
            while(false && a[first:0:last]==a) { preserved=false; }
            for int i in [1:0] { preserved=(a[i/zero:0:999] << -1)==a; }
            if(true || a[first:last]!=a[first:first]) {} else { preserved=false; }
        ''',tuple((name,None) for name in ('x','y','z','preserved','first','last','bad')),(0,1,0,1,0,3,99))

    def test_reached_errors_finish_subtrees_before_comparing_and_do_not_write(self):
        cases = (
            ('a[first:last] == (b[first:last] << (1/zero))',
             ('LEFT_SNAPSHOT_DONE','RIGHT_CHILD_0_SNAPSHOT_DONE'), 'integer division by zero in LET'),
            ('a[first:first] == rotl(b[first:last],1)',
             ('LEFT_SNAPSHOT_DONE','RIGHT_CHILD_0_SNAPSHOT_DONE','RIGHT_DONE'), 'assertion failed in ASSERT'),
            ('a[first:first] == rotl(b[first:last],1/zero)',
             ('LEFT_SNAPSHOT_DONE','RIGHT_CHILD_0_SNAPSHOT_DONE'), 'integer division by zero in LET'),
            ('a[first:99] == (b[first:last] << (1/zero))', (), 'assertion failed in ASSERT'),
            ('(a[first:first] & b[first:last]) == (a[first:last] >> (1/zero))',
             ('LEFT_CHILD_0_SNAPSHOT_DONE','LEFT_CHILD_1_SNAPSHOT_DONE'), 'assertion failed in ASSERT'),
            ('a[first:0:last] != b', (), 'assertion failed in ASSERT'),
            ('a[first:-1:last] == b', (), 'assertion failed in ASSERT'),
            ('a[first:last] == (b << -1)', ('LEFT_SNAPSHOT_DONE',), 'assertion failed in ASSERT'),
        )
        for expression,expected,error in cases:
            with self.subTest(expression=expression):
                program = self.module.openqasm3.parser.parse('OPENQASM 3; bit[4] a="1111"; bit[4] b="0000"; '
                    'int first=0; int last=3; int zero=0; bool result=true; result='+expression+';')
                converter = self.module.QASM2QCXConverter(program)
                converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables,set())
                destination = converter._QASM2QCXConverter__capitalize_variable_name('result')
                lines = []
                labels = ('LEFT_SNAPSHOT_DONE','RIGHT_CHILD_0_SNAPSHOT_DONE','RIGHT_DONE',
                          'LEFT_CHILD_0_SNAPSHOT_DONE','LEFT_CHILD_1_SNAPSHOT_DONE')
                for line in converter:
                    if line.startswith(f'LET {destination} := QASM2QCX_'):
                        lines.append('PRINTLN 999')
                    lines.append(line)
                    for i,label in enumerate(labels):
                        if line == '@QASM2QCX_REGISTER_EQUALITY_0_'+label:
                            lines.append(f'PRINTLN {i+1}')
                lines.append('PRINTLN 888')
                result = self.execute(lines)
                self.assertNotEqual(result.returncode,0)
                self.assertIn(error,result.stderr)
                self.assertNotIn('runtime error:',result.stderr)
                self.assertEqual(result.stdout.split(),[str(labels.index(label)+1) for label in expected])
                type(self).checked += 1

    def test_measurement_feedback(self):
        self.check_exact('''include "stdgates.inc"; qubit[2] q; x q[0];
            bit[2] measured=measure q; int first=0; int last=1;
            if(measured[first:last]=="01") { x q[1]; }
            bit[2] observed=measure q;
            bool feedback=observed[first:last]=="11";
            bool differs=observed[first:last]!=measured[first:last];
        ''',(('feedback',None),('differs',None)),(1,1))

    def test_normal_and_fused_gate_parameters(self):
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = cmath.exp(1j*math.pi/4)
        for fused in (False,True):
            source = '''include "stdgates.inc"; qubit[2] q; h q;
                bit[4] a="1011"; const bit[4] B="1011"; int first=0; int last=3;
                p(pi/4*int(a[first:last]==B[first:last])) q[0];
            '''
            BackendAuditTests.check(self,self.program(source,fused=fused),(0.5,phase/2,0.5,phase/2))


def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bra',required=True,type=pathlib.Path)
    args = parser.parse_args()
    BackendTests.bra = args.bra.resolve()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BackendTests))
    print(f'runtime-range equality programs checked: {BackendTests.checked}; bit-vector comparisons: {BackendTests.comparisons}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
