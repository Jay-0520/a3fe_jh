#!/usr/bin/env python3
"""
Check which repeats (1..5) produce a Boresch restraint, using BioSimSpace in the
same way as a3fe.run.Leg.run_ensemble_equilibration() — but adapted to the given layout:

system_8938/
  bound/
    ensemble_equilibration_1..5/   <-- each has bound_preequil.{prm7,rst7} + gromacs.xtc

For each repeat i:
 - load system from bound/ensemble_equilibration_i/bound_preequil.prm7/rst7
 - decouple ligand (first molecule)
 - run RestraintSearch.analyse(method="BSS", ...)
 - report PASS/FAIL/SKIP

Optional: we try to read append_to_ligand_selection from any
  system_preparation_config_bound.pkl found in repeat dir, else in top-level input/.
"""

import argparse
import glob
import os
import pickle
import sys

import BioSimSpace as BSS
import BioSimSpace.Sandpit.Exscientia as _BSS


def _maybe_load_append_sel(paths):
    """Return append_to_ligand_selection if found in any pickle path, else None."""
    for p in paths:
        try:
            with open(p, "rb") as fh:
                obj = pickle.load(fh)
            if hasattr(obj, "append_to_ligand_selection"):
                return getattr(obj, "append_to_ligand_selection")
        except Exception:
            pass
    return None


def check_repeat(repeat_dir, force_constant=20.0):
    """Return (status, msg) for one repeat dir."""
    temp_k = 298.15
    prm = os.path.join(repeat_dir, "bound_preequil.prm7")
    rst = os.path.join(repeat_dir, "bound_preequil.rst7")
    xtc = os.path.join(repeat_dir, "gromacs.xtc")

    if not (os.path.isfile(prm) and os.path.isfile(rst) and os.path.isfile(xtc)):
        return ("SKIP", "missing bound_preequil.prm7/rst7 or gromacs.xtc")

    try:
        # Load system from this repeat’s own topology/coords (matches your layout)
        system = _BSS.IO.readMolecules([prm, rst])
        # Decouple the first molecule (ligand), like in a3fe
        lig = _BSS.Align.decouple(system[0], intramol=True)
        system.updateMolecule(0, lig)

        traj = _BSS.Trajectory.Trajectory(topology=prm, trajectory=xtc, system=system)

        restraint = _BSS.FreeEnergy.RestraintSearch.analyse(
            method="BSS",
            system=system,
            traj=traj,
            work_dir=repeat_dir,  # scratch here is fine
            temperature=temp_k * _BSS.Units.Temperature.kelvin,
            append_to_ligand_selection="",
            force_constant=(force_constant * _BSS.Units.Energy.kcal_per_mol / (_BSS.Units.Length.angstrom ** 2)),
        )
        if restraint is None:
            return ("FAIL", "analyse() returned None (no stable restraint found)")
        return ("PASS", "restraint found")
    except Exception as e:
        return ("FAIL", f"exception: {e}")


def main():
    ap = argparse.ArgumentParser(description="Test which repeats yield Boresch restraints (BioSimSpace).")
    ap.add_argument("--bound_dir", required=True,
                    help="Path to the bound directory that contains ensemble_equilibration_1..5/")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=5)
    ap.add_argument("--force_constant", type=float, default=10.0,
                    help="Force constant (k) to use in analyse(), default 10.0 kcal/(mol*Å²)")
    args = ap.parse_args()

    bound_dir = os.path.abspath(args.bound_dir)
    if not os.path.isdir(bound_dir):
        print(f"[ERROR] {bound_dir} not found.", file=sys.stderr)
        sys.exit(2)

    print("\n=== Restraint extraction check ===")
    print(f"bound_dir: {bound_dir}")
    print(f"repeats  : {args.start}..{args.end}")

    results = []
    for i in range(args.start, args.end + 1):
        rep_dir = os.path.join(bound_dir, f"ensemble_equilibration_{i}")
        status, msg = check_repeat(repeat_dir=rep_dir, force_constant=args.force_constant)
        print(f"[{i}] {status:<5} - {msg}")
        results.append((i, status, msg))

    print("\n=== Summary ===")
    for i, s, m in results:
        print(f"Repeat {i}: {s} - {m}")

    # exit 1 if any FAIL (so you can use in CI / scripts)
    if any(s == "FAIL" for _, s, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
