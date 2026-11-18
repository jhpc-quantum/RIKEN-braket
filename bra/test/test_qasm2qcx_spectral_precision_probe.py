#!/usr/bin/env python3
"""Compile and smoke-test the separate C++ phase-precision probe."""

import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SpectralPrecisionProbeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('g++'), 'g++ is unavailable')
    def test_probe_runs_for_three_cpp_types(self):
        with tempfile.TemporaryDirectory(prefix='braket-spectral-probe-') as directory:
            executable = Path(directory) / 'probe'
            subprocess.run(['g++', '-std=c++14', '-O2', '-I', str(ROOT / 'ket/include'),
                            str(ROOT / 'bra/test/spectral_precision_backend_probe.cpp'), '-o', str(executable)],
                           check=True, capture_output=True, text=True, timeout=90)
            output = subprocess.run([str(executable)], check=True, capture_output=True, text=True,
                                    timeout=30).stdout
        metadata = re.findall(r'^type=(float|double|long double) radix=(\d+) digits=(\d+)', output, re.MULTILINE)
        self.assertEqual([name for name, _, _ in metadata], ['float', 'double', 'long double'])
        self.assertTrue(all(int(radix) >= 2 and int(digits) >= 2 for _, radix, digits in metadata))
        results = re.findall(r'^(endpoint|below-cut|above-cut|near-degenerate|large-phase) '
                             r'parsed=(\S+) parse_error=(\S+) exp_i_error=(\S+)$', output, re.MULTILINE)
        self.assertEqual(len(results), 15)
        for _, parsed, parsing, execution in results:
            with self.subTest(parsed=parsed, parsing=parsing, execution=execution):
                self.assertTrue(all(math.isfinite(float(value)) for value in (parsed, parsing, execution)))
                self.assertGreaterEqual(float(parsing), 0)
                self.assertLess(float(execution), 1e-5)


if __name__ == '__main__':
    unittest.main()
