#!/usr/bin/env python3

"""Compare fused and unfused execution across Bra MPI state implementations.

Example:
  python3 bra/test/fused_parallel_numerical.py \
    --bra bra/bin/bra --launcher-args="--bind-to none"
"""

import argparse
import math
import pathlib
import re
import shlex
import subprocess
import tempfile


AMPLITUDE_PATTERN = re.compile(
    r"^([01]+) => "
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?) \+ "
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?) i$"
)

CASES = {
    # Four fused qubits in simple mode and three in unit mode leave enough
    # independent outer blocks to occupy three worker threads.
    "outer": (
        "H 7",
        "CU1 7 2 0.375",
        "SWAP 6 5",
    ),
    # All local qubits participate, so the fused callback has one outer block.
    "inner": (
        "H 7",
        "CU1 7 2 0.375",
        "H 6",
        "CU1 6 1 -0.25",
        "SWAP 5 4",
        "H 3",
    ),
    # S exercises the synchronized serial fallback between parallel gates.
    "mixed": (
        "H 7",
        "X 6",
        "eX 6 0.125",
        "eXX 6 5 0.0625",
        "CeX 7 6 -0.09375",
        "sX 6",
        "CU1 7 2 0.375",
        "Y 5",
        "eY 5 -0.1875",
        "eYY 5 3 0.15625",
        "CeY 7 5 0.21875",
        "sY 5",
        "Z 3",
        "eZ 3 0.3125",
        "eZZ 3 4 -0.28125",
        "CeZ 7 3 0.34375",
        "eZ 1 -0.109375",
        "CeZ 7 1 0.140625",
        "sZ 3",
        "sX+ 6",
        "eX+ 6 0.125",
        "eXX+ 6 5 0.0625",
        "CeX+ 7 6 -0.09375",
        "sY+ 5",
        "eY+ 5 -0.1875",
        "eYY+ 5 3 0.15625",
        "CeY+ 7 5 0.21875",
        "sZ+ 3",
        "eZ+ 3 0.3125",
        "eZZ+ 3 4 -0.28125",
        "CeZ+ 7 3 0.34375",
        "eZ+ 1 -0.109375",
        "CeZ+ 7 1 0.140625",
        "S 4",
        "SWAP 4 3",
    ),
}

CONFIGURATIONS = (
    ("simple-nonpage", 4, 6, ("--mode", "simple", "--page-qubits", "0")),
    ("simple-paged", 4, 6, ("--mode", "simple", "--page-qubits", "2")),
    (
        "unit-nonpage",
        3,
        5,
        (
            "--mode",
            "unit",
            "--unit-qubits",
            "3",
            "--unit-processes",
            "3",
            "--page-qubits",
            "0",
        ),
    ),
    (
        "unit-paged",
        3,
        5,
        (
            "--mode",
            "unit",
            "--unit-qubits",
            "3",
            "--unit-processes",
            "3",
            "--page-qubits",
            "2",
        ),
    ),
)


def make_input(gates, fused):
    lines = [
        "QUBITS 8",
        "INITIAL STATE 255",
        "BIT ASSIGNMENT 7 6 5 4 3 2 1 0",
    ]
    if fused:
        lines.append("BEGIN FUSION")
    lines.extend(gates)
    if fused:
        lines.append("END FUSION")
    lines.append("DO AMPLITUDES")
    return "\n".join(lines) + "\n"


def parse_amplitudes(output):
    result = {}
    for line in output.splitlines():
        match = AMPLITUDE_PATTERN.match(line)
        if match is not None:
            result[match.group(1)] = complex(float(match.group(2)), float(match.group(3)))
    return result


def run_bra(args, filename, num_processes, num_cache_qubits, configuration_args):
    command = (
        shlex.split(args.launcher)
        + shlex.split(args.launcher_args)
        + [
            "-n",
            str(num_processes),
            str(args.bra),
            "--file",
            str(filename),
            "--threads",
            str(args.threads),
            "--num-cache-qubits",
            str(num_cache_qubits),
        ]
        + list(configuration_args)
    )
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "command failed: {}\nstdout:\n{}\nstderr:\n{}".format(
                shlex.join(command), error.stdout, error.stderr
            )
        ) from error


def compare_case(args, temporary_directory, configuration, case_name, gates):
    configuration_name, num_processes, num_cache_qubits, configuration_args = configuration
    prefix = temporary_directory / "{}_{}".format(configuration_name, case_name)
    unfused_filename = prefix.with_name(prefix.name + "_unfused.qcx")
    fused_filename = prefix.with_name(prefix.name + "_fused.qcx")
    unfused_filename.write_text(make_input(gates, False), encoding="ascii")
    fused_filename.write_text(make_input(gates, True), encoding="ascii")

    reference = parse_amplitudes(
        run_bra(
            args,
            unfused_filename,
            num_processes,
            num_cache_qubits,
            configuration_args,
        ).stdout
    )
    actual = parse_amplitudes(
        run_bra(
            args,
            fused_filename,
            num_processes,
            num_cache_qubits,
            configuration_args,
        ).stdout
    )

    if len(reference) != 256 or reference.keys() != actual.keys():
        raise RuntimeError(
            "{} {} produced incomplete amplitudes: unfused={}, fused={}".format(
                configuration_name, case_name, len(reference), len(actual)
            )
        )

    maximum_error = max(abs(actual[index] - reference[index]) for index in reference)
    if not math.isfinite(maximum_error) or maximum_error > args.tolerance:
        raise RuntimeError(
            "{} {} failed: max error = {}".format(
                configuration_name, case_name, maximum_error
            )
        )
    print("{} {} passed (max error = {})".format(configuration_name, case_name, maximum_error))


def main():
    repository_root = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--bra", type=pathlib.Path, default=repository_root / "bra/bin/bra")
    parser.add_argument("--launcher", default="mpiexec")
    parser.add_argument("--launcher-args", default="")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()

    if args.threads < 3:
        parser.error("--threads should be at least 3 to exercise both dispatch paths")
    if not args.bra.is_file():
        parser.error("Bra executable does not exist: {}".format(args.bra))

    with tempfile.TemporaryDirectory(prefix="bra-fused-parallel-") as directory:
        temporary_directory = pathlib.Path(directory)
        for configuration in CONFIGURATIONS:
            for case_name, gates in CASES.items():
                compare_case(args, temporary_directory, configuration, case_name, gates)

    print("Bra fused parallel numerical tests passed")


if __name__ == "__main__":
    main()
