"""
quickly check parameterization works for ligand and protein

we could run the following command to directly test the "tleap error"
cat > tleap_min.in <<'EOF'
source leaprc.protein.ff14SB
prot = loadpdb protein.pdb
saveamberparm prot prot.prm7 prot.rst7
quit
EOF
tleap -f tleap_min.in | tee leap_min.log
"""
from pathlib import Path
from a3fe.run.system_prep import SystemPreparationConfig, parameterise_input
from a3fe.run.enums import LegType

IN_DIR  = Path("./input")
OUT_DIR = Path("./output")

def run_leg(leg: LegType) -> bool:
    IN_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    # quick existence checks
    need = ["ligand.sdf"] + (["protein.pdb"] if leg == LegType.BOUND else [])
    for f in need:
        p = IN_DIR / f
        if not p.exists():
            print(f"✗ missing required file: {p}")
            return False

    # minimal config & save (required by parameterise_input)
    cfg = SystemPreparationConfig()
    cfg.save_pickle(str(IN_DIR), leg)

    try:
        print(f"\n=== {leg.name} parameterization ===")
        sys = parameterise_input(leg_type=leg, input_dir=str(IN_DIR), output_dir=str(OUT_DIR))
        # basic success summary
        print("✓ success")
        print(f"  molecules: {sys.nMolecules()}, atoms: {sys.nAtoms()}")
        base = f"{leg.name.lower()}_param"
        for ext in ("prm7", "rst7"):
            f = OUT_DIR / f"{base}.{ext}"
            print(f"  {'✓' if f.exists() else '✗'} {f}")
        return True
    except Exception as e:
        print("✗ failed")
        print(f"  error: {e}")
        return False

def main():
    print("A3FE parameterization quick check\n")
    # Run BOUND (protein + ligand)
    ok_bound = run_leg(LegType.BOUND)
    print("\n=== summary ===")
    print(f"BOUND: {'PASS' if ok_bound else 'FAIL'}")

if __name__ == "__main__":
    main()