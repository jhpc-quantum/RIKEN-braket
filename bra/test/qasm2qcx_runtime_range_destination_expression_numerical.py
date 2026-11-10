#!/usr/bin/env python3
"""Audit runtime-range expression assignments into runtime ranges (#521).

Run with --bra /path/to/bra (a non-MPI build, including UBSan builds).
Independent bit vectors check complete destination registers, including untouched
bits. Failure markers check validation order and suppression of writes.
"""

import argparse
import cmath
import math
import pathlib
import resource
import unittest
from unittest.mock import patch

import qasm2qcx_runtime_range_equality_numerical as registers
import qasm2qcx_runtime_set_expression_assignment_numerical as index_sets
from qasm2qcx_static_selection_expression_assignment_numerical import write_selected


class BackendTests(unittest.TestCase):
    bra = None
    checked = 0
    assignments = 0
    setUp = registers.BackendTests.setUp
    execute = registers.BackendTests.execute
    program = registers.BackendTests.program
    check_exact = registers.BackendTests.check_exact

    # These generators depend on self.check_nodes only for destination handling.
    # Reuse their source patterns and independent expression-tree references.
    check_case = index_sets.BackendTests.check_case
    test_exhaustive_small_patterns_and_source_kinds = index_sets.BackendTests.test_exhaustive_small_patterns_and_source_kinds
    test_native_widths_wide_sources_and_negative_ranges = index_sets.BackendTests.test_native_widths_wide_sources_and_negative_ranges
    test_deep_trees_and_native_endpoint_distances = index_sets.BackendTests.test_deep_trees_and_native_endpoint_distances

    def check_nodes(self, source, nodes, before, preservation=()):
        width = len(nodes[0][1])
        size = len(before)
        self.assertEqual(size,2*width+1)
        declarations = (f'int dstart=0; uint dstop={width-1}; uint high={size-1}; '
                        f'int stride=-2; uint low=1; array[int,3] bounds={{-{size-1},2,-2}};')
        selections = (('dstart:dstop',list(range(width))),
                      ('high:stride:low',list(range(size-1,1,-2))),
                      ('bounds[0]:bounds[1]:bounds[2]',list(range(1,size-1,2))))
        statements,outputs,expected = [source,declarations],[],[]
        for selector,positions in selections:
            for expression,bits in nodes:
                name = f'out{len(outputs)}'
                statements.extend((f'bit[{size}] {name}={registers.literal(before)};',
                                   f'{name}[{selector}]={expression};'))
                result = write_selected(before,positions,bits)
                outputs.append((name,size))
                expected.append(sum(bit << i for i,bit in enumerate(result)))
        for i,(expression,value) in enumerate(preservation):
            name = f'preserved{i}'
            statements.append(f'bool {name}={expression};')
            outputs.append((name,None))
            expected.append(value)
        self.check_exact(' '.join(statements),outputs,expected)
        type(self).assignments += len(nodes)*len(selections)

    def test_self_overlap_repeated_sources_and_bounds_read_from_original_destination(self):
        cases = (((0,1,3),(3,-1,0)),((-1,-1,-4),(-4,1,-1)),
                 ((-1,4,3),(3,-3,0)),((0,2,3),(-2,3,1)))
        for pattern in range(16):
            before = [(pattern >> i)&1 for i in range(4)]
            statements,outputs,expected = [],[],[]
            for case,(source_bounds,destination_bounds) in enumerate(cases):
                first,step,last = source_bounds
                start,stride,stop = destination_bounds
                positions = list(range(start,stop+(1 if stride>0 else -1),stride))
                positions = [position%4 for position in positions]
                bits = registers.selected(pattern,4,source_bounds)
                statements.append(f'int first{case}={first}; int step{case}={step}; int last{case}={last}; '
                                  f'int start{case}={start}; int stride{case}={stride}; int stop{case}={stop};')
                for operation in ('~','rotl','^'):
                    name = f'out{len(outputs)}'
                    leaf = f'{name}[first{case}:step{case}:last{case}]',bits
                    if operation == '^':
                        mask = [i%2 for i in range(len(bits))]
                        node = registers.binary(leaf,(registers.literal(mask),mask),'^')
                    elif operation == 'rotl':
                        node = registers.unary(leaf,'rotl',1,f'int({name}=={registers.literal(before)})')
                    else:
                        node = registers.unary(leaf,'~')
                    statements.extend((f'bit[4] {name}={registers.literal(before)};',
                                       f'{name}[start{case}:stride{case}:stop{case}]={node[0]};'))
                    result = write_selected(before,positions,node[1])
                    outputs.append((name,4))
                    expected.append(sum(bit << i for i,bit in enumerate(result)))
            self.check_exact(' '.join(statements),outputs,expected)
            type(self).assignments += len(outputs)
        self.check_exact('''bit[4] out="0000"; const bit[2] Z="00"; int first=0; int last=1;
            out[int(out[0]):int(out[1])+1]=~Z[first:last];
        ''',(('out',4),),(3,))

    def test_native_endpoint_steps_and_unselected_stop_bounds(self):
        c = self.module.QASM2QCXConverter
        cases = (('int',0,'uint',c.QCX_UINT_MAX,'uint',c.QCX_UINT_MAX-1),
                 ('int',-1,'int',c.QCX_INT_MIN,'int',c.QCX_INT_MIN),
                 ('int',0,'int',c.QCX_INT_MAX,'int',c.QCX_INT_MAX-1),
                 ('uint',3,'int',c.QCX_INT_MIN,'uint',0),
                 ('int',-1,'uint',c.QCX_UINT_MAX,'int',3))
        for start_type,start,step_type,step,stop_type,stop in cases:
            with self.subTest(start=start,step=step,stop=stop):
                positions = [position%4 for position in range(start,stop+(1 if step>0 else -1),step)]
                before = [0,1,0,1]
                result = write_selected(before,positions,[1])
                token = 'uint(-1)' if step == c.QCX_UINT_MAX else str(step)
                self.check_exact(f'''bit[4] out="1010"; const bit[1] Z="0"; int zero=0;
                    {start_type} start={start}; {step_type} step={token}; {stop_type} stop={stop};
                    out[start:step:stop]=~Z[zero:zero];
                ''',(('out',4),),(sum(bit << i for i,bit in enumerate(result)),))
        self.check_exact('''bit[1] one="1"; int first=-1; uint last=0;
            one[first:first]=~one[last:last];
        ''',(('one',1),),(0,))

    def test_loop_reentry_refreshes_both_ranges_and_source_values(self):
        source = [(11 >> i)&1 for i in range(4)]
        before = [(85 >> i)&1 for i in range(8)]
        total = 0
        for positions in ((7,5,3,1),(6,4,2,0),(0,2,4,6)):
            source[0] = 1-source[0]
            bits = [bit ^ ((3 >> i)&1) for i,bit in enumerate(source)]
            before = write_selected(before,positions,bits)
            total += sum(bit << i for i,bit in enumerate(before))
        self.check_exact('''bit[4] a="1011"; bit[8] out="01010101";
            int first=0; int last=3; int high=7; int stride=-2; int low=1; int total=0;
            for int i in [0:2] {
                a[0]=!a[0]; out[high:stride:low]=a[first:last]^"0011";
                total+=int(uint[8](out));
                if(i==0) { high=6; low=0; }
                else { high=0; low=6; stride=2; }
            }
        '''+f'bool fresh=out=={registers.literal(before)};',(('total',None),('fresh',None)),(total,1))
        before = [(85 >> i)&1 for i in range(8)]
        values = []
        for distance,(bounds,positions) in enumerate(zip(((0,1,3),(1,2,7),(7,-2,1)),
                                                        ((7,5,3,1),(6,4,2,0),(0,2,4,6)))):
            node = registers.unary(registers.unary(('',registers.selected(180,8,bounds)),'rotl',distance),'~')
            before = write_selected(before,positions,node[1])
            values.append(sum(bit << i for i,bit in enumerate(before)))
        self.check_exact('''bit[8] source="10110100"; bit[8] out="01010101";
            int first=0; int step=1; int last=3; int visits=0; int high=7; int stride=-2; int low=1;
            array[uint[8],3] values={0,0,0};
            while(visits<3) {
                out[high:stride:low]=~rotl(source[first:step:last],visits);
                values[visits]=uint[8](out);
                if(visits==0) { first=1; step=2; last=7; high=6; low=0; }
                else { first=7; step=-2; last=1; high=0; stride=2; low=6; }
                visits+=1;
            }
            uint first_value=values[0]; uint second_value=values[1]; uint third_value=values[2];
            bool unchanged=source=="10110100";
        ''',tuple((name,None) for name in ('first_value','second_value','third_value','unchanged')),(*values,1))

    def test_nested_scalar_consumers_short_circuiting_and_skipped_domains(self):
        before = [(85 >> i)&1 for i in range(8)]
        bits = registers.unary(('',[1,1,0,1]),'rotl',2)[1]
        result = write_selected(before,(7,5,3,1),bits)
        self.check_exact('''bit[4] a="1011"; const bit[4] B="1101"; bit[8] out="01010101";
            int first=0; int last=3; array[int,2] endpoints={7,1};
            out[endpoints[0]:-2:endpoints[1]]=rotl(a[first:last],
                popcount(a[first:last]&B)+int(a[first:last]==B[first:last]));
        ''',(('out',8),('a',4)),(sum(bit << i for i,bit in enumerate(result)),11))
        self.check_exact('''bit[4] a="1011"; const bit[4] B="1101"; bit[8] out="01010101";
            int first=0; int last=3; int zero=0;
            if(false) { out[first/zero:0:999]=rotl(a[first/zero:0:999],1/zero); }
            while(false) { out[-1:first:7]=a[first:0:last]<<-1; }
            for int i in [1:0] { out[i/0:0:999]=~a[i/0:0:999]; }
            bool preserved=out=="01010101";
            out[int(false && a[first:0:last]==B):last]=rotl(a[first:last],int(true || a[first:0:last]==B));
            bool changed=out=="01010111";
        ''',(('preserved',None),('changed',None)),(1,1))

    def test_reached_failures_precede_writes_and_destination_errors_precede_rhs(self):
        cases = (
            ('first:0:last','rotl(a[first:last],1/zero)','assertion failed in ASSERT',False,None),
            ('last:1:first','rotl(a[first:last],1/zero)','assertion failed in ASSERT',False,None),
            ('first:1:8','rotl(a[first:last],1/zero)','assertion failed in ASSERT',False,None),
            ('-1:8:last+4','rotl(a[first:last],1/zero)','assertion failed in ASSERT',False,None),
            ('first:0:1/zero','~a[first:last]','integer division by zero in LET',False,None),
            ('first:1:1','rotl(a[first:last],1/zero)','integer division by zero in LET',False,None),
            ('first:1:1','~a[first:last]','assertion failed in ASSERT',True,None),
            ('first:1:1','(a[first:first]&B)^rotl(a[first:last],1/zero)','assertion failed in ASSERT',False,None),
            ('first:1:1','~a[first:0:last]','assertion failed in ASSERT',False,None),
            ('first:1:1','a[first:last]>>-1','assertion failed in ASSERT',False,None),
            ('first:1:1','a[first:99]^B','assertion failed in ASSERT',False,None),
            ('first:1:last','rotl(a[first:last],1/zero)','assertion failed in ASSERT',False,2),
            ('first:first','~a[first:last]','assertion failed in ASSERT',False,2),
        )
        for bounds,rhs,error,complete,capacity in cases:
            with self.subTest(bounds=bounds,rhs=rhs,capacity=capacity):
                source = ('bit[4] a="1011"; const bit[4] B="1101"; bit[8] out="01010101"; '
                          'int first=0; int last=3; int zero=0; '+f'out[{bounds}]={rhs};')
                program = self.module.openqasm3.parser.parse('OPENQASM 3; '+source)
                converter = self.module.QASM2QCXConverter(program)
                if capacity is None:
                    converter.visit(program)
                else:
                    with patch.object(converter,'MAX_RUNTIME_BIT_RANGE_POSITIONS',capacity):
                        converter.visit(program)
                self.assertEqual(converter._QASM2QCXConverter__used_temporary_variables,set())
                modified = []
                for line in converter:
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

    def test_later_iteration_length_or_duplicate_failure_prevents_partial_writes(self):
        for update in ('low=3;','high=-1; stride=8; low=7;'):
            lines = self.program('''bit[4] a="1011"; bit[8] out="01010101"; int first=0; int last=3;
                int high=7; int stride=-2; int low=1; int visits=0;
                while(visits<2) { out[high:stride:low]=~a[first:last]; '''+update+' visits+=1; }')
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
            int high=5; out[high:-1:high-1]=rotl(measured[first:last],1);
            if(out[4]) { x q[1]; }
            bit[2] observed=measure q; bool feedback=observed=="11";
        ''',(('out',8),('feedback',None)),(16,1))
        from qasm2qcx_u_power_numerical import BackendAuditTests
        phase = cmath.exp(1j*math.pi/4)
        for fused in (False,True):
            source = '''include "stdgates.inc"; qubit[2] q; h q;
                bit[4] a="1011"; bit[8] out="00000000"; int first=0; int last=3;
                out[last+4:-2:first+1]=rotl(a[first:last],1);
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
    print(f'runtime-range destination expression programs checked: {BackendTests.checked}; destination registers checked: {BackendTests.assignments}')
    raise SystemExit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
