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


def native_principal_argument(value):
    """Match the backend's negative-real endpoint rule for a computed value."""
    if not (math.isfinite(value.real) and math.isfinite(value.imag)) or value == 0:
        raise ValueError('a finite, nonzero eigenvalue is required')
    if value.real < 0 and value.imag == 0:
        return math.pi
    return cmath.phase(value)


def spectral_power_matrix(unitary, exponents):
    """Model the existing U-family eigenphase path using host floating point.

    This is an experiment, not an implementation of the QCX backend. The two
    eigenvalues are selected separately before each power; no scalar snapping
    or quaternion normalization is performed.
    """
    x, y, z = unitary.vector
    length = math.hypot(x, y, z)
    plus = unitary.phase * complex(unitary.scalar, -length)
    minus = unitary.phase * complex(unitary.scalar, length)
    for exponent in exponents:
        if not math.isfinite(exponent):
            raise ValueError('power must be finite')
        plus = cmath.exp(1j * exponent * native_principal_argument(plus))
        minus = cmath.exp(1j * exponent * native_principal_argument(minus))
    common = (plus + minus) / 2
    if length == 0:
        return ((common, 0j), (0j, common))
    difference = (plus - minus) / 2
    nx, ny, nz = x / length, y / length, z / length
    return ((common + difference * nz, difference * complex(nx, -ny)),
            (difference * complex(nx, ny), common - difference * nz))


if __name__ == '__main__':
    # Four T gates are exactly Z, whose principal square root is S. Report the
    # unsnapped result: host rounding may put its -1 eigenvalue below the cut.
    for names in (('s', 's'), ('t', 't', 't', 't')):
        unitary = compose_sequence(names)
        print(' '.join(names), 'sqrt:', spectral_power_matrix(unitary, (0.5,)))
