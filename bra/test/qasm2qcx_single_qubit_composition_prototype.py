#!/usr/bin/env python3
"""Experimental composition of fixed single-qubit gates; not converter code.

The representation is c * (a I - i v.sigma), with complex c and real a, v.
This prototype deliberately does not normalize or snap small values to zero.
"""

import cmath
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Unitary:
    phase: complex
    scalar: float
    vector: tuple[float, float, float]


IDENTITY = Unitary(1 + 0j, 1.0, (0.0, 0.0, 0.0))


def axis_phase(axis, phase):
    """Return a gate with eigenvalues 1 and exp(i*phase) on a Pauli axis."""
    half = phase / 2
    sine = math.sin(half)
    return Unitary(cmath.exp(1j * half), math.cos(half),
                   tuple(sine * component for component in axis))


def fixed_gates():
    inv_sqrt_two = math.sqrt(0.5)
    x, y, z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    # Explicit zero scalars preserve the exact self-adjoint gate matrices;
    # cos(pi/2) in floating point would introduce an artificial component.
    return {
        'id': IDENTITY,
        'x': Unitary(1j, 0.0, x),
        'y': Unitary(1j, 0.0, y),
        'z': Unitary(1j, 0.0, z),
        'h': Unitary(1j, 0.0, (inv_sqrt_two, 0.0, inv_sqrt_two)),
        's': axis_phase(z, math.pi / 2),
        'sdg': axis_phase(z, -math.pi / 2),
        't': axis_phase(z, math.pi / 4),
        'tdg': axis_phase(z, -math.pi / 4),
        'sx': axis_phase(x, math.pi / 2),
    }


def compose(after, before):
    """Return after * before; both arguments must be Unitary values."""
    a, b = after.scalar, before.scalar
    x, y, z = after.vector
    u, v, w = before.vector
    return Unitary(
        after.phase * before.phase,
        a * b - (x * u + y * v + z * w),
        (a * u + b * x + y * w - z * v,
         a * v + b * y + z * u - x * w,
         a * w + b * z + x * v - y * u))


def compose_sequence(names):
    gates = fixed_gates()
    result = IDENTITY
    for name in names:
        result = compose(gates[name], result)
    return result


def as_matrix(unitary):
    a = unitary.scalar
    x, y, z = unitary.vector
    c = unitary.phase
    return ((c * complex(a, -z), c * complex(-y, -x)),
            (c * complex(y, -x), c * complex(a, z)))
