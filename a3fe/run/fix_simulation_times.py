"""
This is very messy but support the following features:

"""
import os
import glob
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Tuple, List

import numpy as np

from a3fe.run.enums import LegType as _LegType, StageType as _StageType


# =========================
# Basic helpers (no dups)
# =========================
def _find_window(calc, leg_name, stage_name, lam):
    leg = next((L for L in calc.legs if L.leg_type.name.lower() == str(leg_name).lower()), None)
    if not leg:
        return None
    stage = next((S for S in leg.stages if S.stage_type.name.lower() == str(stage_name).lower()), None)
    if not stage:
        return None
    win = next((W for W in stage.lam_windows if abs(W.lam - float(lam)) < 1e-9), None)
    if not win:
        return None
    return leg, stage, win


def _split_header_and_data(lines):
    header, data = [], []
    for ln in lines:
        if ln.startswith("#") or not ln.strip():
            header.append(ln)
        else:
            data.append(ln)
    return header, data


def _parse_step_from_line(line):
    s = line.strip()
    if not s:
        return None
    parts = s.split()
    try:
        return int(parts[0])
    except Exception:
        return None


def _replace_step(line: str, new_step: int) -> str:
    """Replace only the first column with new_step, preserve spacing and trailing spaces."""
    idx = line.find(" ")
    if idx == -1:
        return f"{new_step}\n"
    return f"{new_step}{line[idx:]}"


# =========================
# Reset / deletion utility
# =========================
def delete_checkpoints_for_lambda(calc, *, leg, stage, lam,
                                  delete_simfile=True,
                                  delete_gradients_dat=True,
                                  extra_patterns=None):
    """
    Minimal reset for a single λ-window:
      - delete *.s3 (SYSTEM/gradients/sim_restart, etc.)
      - optionally delete simfile.dat and gradients.dat so fresh files are written

    NOTE: This function does NOT modify/trim gradients.dat; use backup+reset below
          if you want a safe copy first.
    """
    logger = calc._logger
    found = _find_window(calc, leg, stage, lam)
    if not found:
        logger.error(f"[reset] λ-window not found: leg={leg}, stage={stage}, λ={lam}")
        return False

    _, _, win = found
    patterns = ["*.s3", "*.s3.previous"] # catches SYSTEM.s3, gradients.s3, sim_restart.s3, etc.
    if extra_patterns:
        patterns.extend(extra_patterns)

    ok = True
    for sim in win.sims:
        run_dir = Path(sim.output_dir)

        # remove *.s3 (and extras)
        for pat in patterns:
            for p in run_dir.glob(pat):
                try:
                    p.unlink()
                    logger.info(f"[reset] removed {p}")
                except Exception as e:
                    ok = False
                    logger.warning(f"[reset] could not remove {p}: {e}")

        # optionally drop simfile.dat so a fresh file is written (no new gaps)
        if delete_simfile:
            simfile = run_dir / "simfile.dat"
            if simfile.exists():
                try:
                    simfile.unlink()
                    logger.info(f"[reset] removed {simfile}")
                except Exception as e:
                    ok = False
                    logger.warning(f"[reset] could not remove {simfile}: {e}")

        # optionally drop gradients.dat (we do NOT edit it elsewhere)
        if delete_gradients_dat:
            fp = run_dir / "gradients.dat"
            if fp.exists():
                try:
                    fp.unlink()
                    logger.info(f"[reset] removed {fp}")
                except Exception as e:
                    ok = False
                    logger.warning(f"[reset] could not remove {fp}: {e}")

    return ok


# =========================================
# Time reading & gap-aware measurements
# =========================================
def _get_actual_simtime_from_file(sim, timestep_ns=4e-6, detect_gaps=True):
    """
    Determine actual time from simfile.dat: last valid data step,
    or end of first contiguous block if detect_gaps=True.
    """
    simfile_path = os.path.join(sim.output_dir, "simfile.dat")
    if not os.path.exists(simfile_path) or os.stat(simfile_path).st_size == 0:
        return 0.0

    steps = []
    with open(simfile_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            try:
                step = int(line.split()[0])
                steps.append(step)
            except Exception:
                continue

    if not steps:
        return 0.0
    if not detect_gaps or len(steps) < 2:
        return steps[-1] * timestep_ns

    intervals = np.diff(steps)
    median_interval = np.median(intervals) if len(intervals) else 0
    gap_threshold = 2 * median_interval if median_interval > 0 else 0
    large_gaps = np.where(intervals > gap_threshold)[0] if gap_threshold > 0 else []

    if len(large_gaps) == 0:
        return steps[-1] * timestep_ns

    end_idx = large_gaps[0]  # last sample before first gap
    return steps[end_idx] * timestep_ns


def _get_first_block_end_time(sim, timestep_ns=None, gap_factor=2.0):
    """Return (end_time_ns, had_gap, details) for the first contiguous block."""
    simfile_path = os.path.join(sim.output_dir, "simfile.dat")
    if not os.path.exists(simfile_path) or os.stat(simfile_path).st_size == 0:
        return 0.0, False, {}

    if timestep_ns is None:
        timestep_ns = getattr(sim, "timestep", 4e-6)

    steps = []
    with open(simfile_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            try:
                steps.append(int(line.split()[0]))
            except Exception:
                continue

    if not steps:
        return 0.0, False, {}
    if len(steps) < 2:
        return steps[-1] * timestep_ns, False, {"start_step": steps[0], "end_step": steps[-1], "block_len": 1}

    intervals = np.diff(steps)
    median_interval = np.median(intervals) if len(intervals) else 0
    if median_interval <= 0:
        return steps[-1] * timestep_ns, False, {"start_step": steps[0], "end_step": steps[-1], "block_len": len(steps)}

    large_gaps = np.where(intervals > gap_factor * median_interval)[0]
    if len(large_gaps) == 0:
        return steps[-1] * timestep_ns, False, {"start_step": steps[0], "end_step": steps[-1], "block_len": len(steps)}

    end_idx = large_gaps[0]
    details = {"start_step": steps[0], "end_step": steps[end_idx], "block_len": end_idx + 1}
    return steps[end_idx] * timestep_ns, True, details


# ============================
# Truncation & verification
# ============================
def truncate_simulation_file(simulation, target_time_ns, logger, detect_gaps=True):
    """
    Truncate simfile.dat to target_time_ns (ns).
    If detect_gaps=True, also reduce to first contiguous block before truncation.
    """
    simfile_path = os.path.join(simulation.output_dir, "simfile.dat")
    if not os.path.exists(simfile_path):
        logger.warning(f"Warning: {simfile_path} does not exist, skipping truncation")
        return

    timestep_ns = getattr(simulation, "timestep", 4e-6)
    with open(simfile_path, "r") as f:
        lines = f.readlines()

    header_lines, steps = [], []
    for line in lines:
        if line.startswith("#"):
            header_lines.append(line)
        elif line.strip():
            try:
                step = int(line.split()[0])
                time_ns = step * timestep_ns
                steps.append((step, time_ns, line))
            except Exception:
                continue

    if not steps:
        logger.warning(f"Warning: No valid data found in {simfile_path}")
        return

    # optional inner gap handling
    if detect_gaps and len(steps) > 1:
        step_nums = [s[0] for s in steps]
        intervals = np.diff(step_nums)
        median_interval = np.median(intervals) if len(intervals) else 0
        if median_interval > 0:
            gap_threshold = 2 * median_interval
            large_gaps = np.where(intervals > gap_threshold)[0]
            if len(large_gaps) > 0:
                end_idx = large_gaps[0]
                logger.warning(f"     Gap detected in {simfile_path} → using first contiguous block")
                steps = steps[: end_idx + 1]

    # time-based truncation
    final_data_lines = []
    for step, time_ns, line in steps:
        if time_ns <= target_time_ns + 1e-9:
            final_data_lines.append(line)
        else:
            break

    if not final_data_lines:
        logger.warning(f"Warning: No valid data found within target time in {simfile_path}")
        return

    # backup once
    backup_path = simfile_path + ".backup"
    if not os.path.exists(backup_path):
        subprocess.run(["cp", simfile_path, backup_path], check=False)

    # write truncated file
    with open(simfile_path, "w") as f:
        for line in header_lines:
            f.write(line)
        for line in final_data_lines:
            f.write(line)

    last_step = int(final_data_lines[-1].split()[0])
    actual_time = last_step * timestep_ns
    logger.info(f"     Truncated to step {last_step}, actual time: {actual_time:.6f} ns")


def truncate_simulations_to_minimum(calc, detect_gaps=True):
    """
    For each λ-window:
      - if any run has a gap → trim each run to its first contiguous block.
      - else if (max - min) > 0.01 ns → truncate longer runs down to min.
    """
    logger = calc._logger

    for leg in calc.legs:
        for stage in leg.stages:
            stage_has_issues = False
            for lam_window in stage.lam_windows:
                sim_times, had_gaps, first_block_times = [], [], []
                for sim in lam_window.sims:
                    fb_time, had_gap, details = _get_first_block_end_time(
                        sim,
                        timestep_ns=getattr(sim, "timestep", 4e-6),
                        gap_factor=2.0,
                    )
                    first_block_times.append(fb_time)
                    had_gaps.append(had_gap)
                    sim_times.append(fb_time)
                    if had_gap:
                        logger.warning(
                            f"  Gap detected in {os.path.join(sim.output_dir, 'simfile.dat')}: "
                            f"keeping first block ({details.get('start_step')}–{details.get('end_step')})"
                        )

                if detect_gaps and any(had_gaps):
                    logger.info(
                        f"❌ {leg.leg_type.name} {stage.stage_type.name} λ={lam_window.lam:.3f}: gaps → trim to first block"
                    )
                    for sim, fb_time in zip(lam_window.sims, first_block_times):
                        truncate_simulation_file(sim, fb_time, logger, detect_gaps=False)
                    continue

                # no gaps → standard consistency cut
                if not sim_times:
                    continue
                min_time = min(sim_times)
                max_time = max(sim_times)

                if abs(max_time - min_time) > 0.01:
                    stage_has_issues = True
                    logger.warning(
                        f"❌ {leg.leg_type.name} {stage.stage_type.name} λ={lam_window.lam:.3f}: "
                        f"Inconsistent times {np.round(sim_times, 6)} → truncating to {min_time:.6f} ns"
                    )
                    for i, sim in enumerate(lam_window.sims):
                        if sim_times[i] > min_time:
                            truncate_simulation_file(sim, min_time, logger, detect_gaps=False)
                else:
                    logger.debug(
                        f"{leg.leg_type.name} {stage.stage_type.name} λ={lam_window.lam:.3f}: "
                        f"✅ consistent at {min_time:.6f} ns"
                    )

            if not stage_has_issues:
                logger.info(f" - {leg.leg_type.name} {stage.stage_type.name}: no timing issues")


def verify_truncation(calc):
    """
    Verify (post-trim) using last available step (with gap detection on so a broken file flags).
    """
    logger = calc._logger
    all_consistent = True

    for leg in calc.legs:
        for stage in leg.stages:
            for lam_window in stage.lam_windows:
                sim_times = []
                for sim in lam_window.sims:
                    t_ns = _get_actual_simtime_from_file(
                        sim,
                        timestep_ns=getattr(sim, "timestep", 4e-6),
                        detect_gaps=True,
                    )
                    sim_times.append(t_ns)

                if not sim_times:
                    continue
                min_time = min(sim_times)
                max_time = max(sim_times)

                if abs(max_time - min_time) > 0.01:
                    logger.error(
                        f"{leg.leg_type.name} {stage.stage_type.name} "
                        f"λ={lam_window.lam:.3f}: ❌ Still inconsistent: {np.round(sim_times, 6)}"
                    )
                    all_consistent = False
                else:
                    logger.debug(
                        f"{leg.leg_type.name} {stage.stage_type.name} "
                        f"λ={lam_window.lam:.3f}: ✅ Consistent at {min_time:.6f} ns"
                    )

    if all_consistent:
        logger.info("✅ ALL SIMULATIONS NOW HAVE CONSISTENT TIMES!")
    else:
        logger.error("❌ Some simulations still have inconsistent times")

    return all_consistent


# ============================================
# Auto-pick λ to restart (simple policy)
# ============================================
def scan_runtime_issues(calc, *, detect_gaps=True, gap_factor=2.0, tol_ns=0.01):
    """
    Return {(leg, stage, lam): {'times': [...], 'had_gaps': [...], 'min','max','range','median','tol_ns'}}
    Times are first-block end if detect_gaps=True.
    """
    out = {}
    for leg in calc.legs:
        leg_name = leg.leg_type.name
        for stage in leg.stages:
            stage_name = stage.stage_type.name
            for win in stage.lam_windows:
                per_times, had_gaps = [], []
                for sim in win.sims:
                    if detect_gaps:
                        t_ns, g, _ = _get_first_block_end_time(
                            sim,
                            timestep_ns=getattr(sim, "timestep", 4e-6),
                            gap_factor=gap_factor,
                        )
                    else:
                        t_ns = _get_actual_simtime_from_file(
                            sim,
                            timestep_ns=getattr(sim, "timestep", 4e-6),
                            detect_gaps=False,
                        )
                        g = False
                    per_times.append(float(t_ns))
                    had_gaps.append(bool(g))

                if per_times:
                    arr = np.array(per_times, dtype=float)
                    out[(leg_name, stage_name, float(win.lam))] = {
                        "times": per_times,
                        "had_gaps": had_gaps,
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                        "range": float(np.max(arr) - np.min(arr)),
                        "median": float(np.median(arr)),
                        "tol_ns": float(tol_ns),
                    }
    return out


def pick_windows_to_restart(scan_result,
                            *,
                            restart_if_gap=True,
                            restart_if_inconsistent=True,
                            tol_ns=0.01,
                            outlier_fraction=0.80):
    """
    Choose λ-windows to restart based on gaps/inconsistency/short-outliers.
    Returns [(leg, stage, lam, reason_dict), ...]
    """
    picks = []
    for (leg, stage, lam), info in scan_result.items():
        reason = {}

        if restart_if_gap and any(info["had_gaps"]):
            reason["gap"] = True

        if restart_if_inconsistent and info["range"] > tol_ns:
            reason["inconsistent"] = {"range_ns": info["range"], "tol_ns": tol_ns}

        med = info["median"]
        if med > 0:
            short_flags = [t < outlier_fraction * med for t in info["times"]]
            if any(short_flags):
                reason["short_outlier"] = {
                    "median_ns": med,
                    "fraction": outlier_fraction,
                    "times": info["times"],
                }

        if reason:
            picks.append((leg, stage, lam, reason))

    def _sev(item):
        _leg, _stage, _lam, r = item
        rng = scan_result[(_leg, _stage, _lam)]["range"]
        gap = 1 if "gap" in r else 0
        outl = 1 if "short_outlier" in r else 0
        return (rng, gap, outl)
    picks = sorted(picks, key=_sev, reverse=True)

    return picks


# =================================================
# One-button workflow with optional auto-restart
# =================================================
def fix_simulation_times(
    calc,
    apply_truncation=True,
    detect_gaps=True,
    *,
    # auto-restart knobs
    apply_auto_restart=True,
    restart_if_gap=True,
    restart_if_inconsistent=True,
    inconsistency_tol_ns=0.01,
    outlier_fraction=0.80,
    delete_simfile_on_restart=True,
    delete_gradientfile_on_restart=True,  # whether to delete simfile.dat on restart
    extra_ckpt_patterns=None,
    dry_run=False,
):
    """
    1) Scan λ-windows, pick problematic ones,
    2) Optionally delete *.s3 (+simfile.dat) for just those λ,
    3) Truncate pass (if requested),
    4) Verify.
    """
    logger = calc._logger

    # 1) Scan & report
    scan = scan_runtime_issues(
        calc, detect_gaps=detect_gaps, gap_factor=2.0, tol_ns=inconsistency_tol_ns
    )
    picks = pick_windows_to_restart(
        scan,
        restart_if_gap=restart_if_gap,
        restart_if_inconsistent=restart_if_inconsistent,
        tol_ns=inconsistency_tol_ns,
        outlier_fraction=outlier_fraction,
    )

    if picks:
        logger.info("=== Auto-restart candidates (based on runtime analysis) ===")
        for leg, stage, lam, reason in picks:
            info = scan[(leg, stage, lam)]
            logger.info(
                f"  λ={lam:.3f} ({leg}/{stage})  times={np.round(info['times'], 6)}  "
                f"range={info['range']:.4f} ns  median={info['median']:.4f} ns  reasons={list(reason.keys())}"
            )
    else:
        logger.info("No λ windows selected for restart by the current policy.")

    # 2) Apply deletions for the selected λ
    if apply_auto_restart and picks:
        for leg, stage, lam, _ in picks:
            if dry_run:
                logger.info(f"[auto-restart:DRY] would reset λ={lam:.3f} ({leg}/{stage})")
            else:
                logger.info(f"[auto-restart] resetting λ={lam:.3f} ({leg}/{stage})")
                delete_checkpoints_for_lambda(
                    calc,
                    leg=leg,
                    stage=stage,
                    lam=lam,
                    delete_simfile=delete_simfile_on_restart,
                    delete_gradients_dat=delete_gradientfile_on_restart,
                    extra_patterns=extra_ckpt_patterns,
                )

    # 3) Truncation pass
    if apply_truncation and not apply_auto_restart and not dry_run:
        logger.info("Starting simulation time fixing process (truncate pass)...")
        truncate_simulations_to_minimum(calc, detect_gaps=detect_gaps)

    # 4) Verify
    success = verify_truncation(calc)
    if success:
        logger.info("\n🎉 Ready to proceed with next steps!")
    else:
        logger.warning("\n⚠️ Some issues remain. Consider adjusting policy or manual inspection.")

    return {"success": success, "restart_picks": picks, "scan": scan}


# ============================================
# Minimal extension helpers (no archiving)
# ============================================
_MARK = ".extend_meta.json"

def _read_sim_lines(path: Path) -> Tuple[List[str], List[Tuple[int, float, str]]]:
    """Return (headers, [(step, time_ns, line), ...]) for a simfile-like file."""
    if not path.exists() or path.stat().st_size == 0:
        return [], []
    headers, data = [], []
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                headers.append(line)
                continue
            parts = line.split()
            try:
                step = int(parts[0])
                time_ns = float(parts[1]) if len(parts) > 1 else np.nan
                data.append((step, time_ns, line))
            except Exception:
                continue
    return headers, data


def _median_step_interval(steps: List[int]) -> int:
    if len(steps) < 2:
        return 0
    diffs = np.diff(np.array(steps, dtype=np.int64))
    diffs = diffs[diffs > 0]
    return int(np.median(diffs)) if len(diffs) else 0


def start_extension(run_dir: str, old_file: str = "simfile.dat", snapshot_name="simfile.preextend.dat"):
    """
    Snapshot current simfile and move it aside so SOMD writes a fresh one.
    """
    run = Path(run_dir)
    old = run / old_file
    snap = run / snapshot_name

    headers, data = _read_sim_lines(old)
    last_step_old = data[-1][0] if data else 0

    if old.exists():
        snap.write_text(old.read_text())
        old.rename(run / (old_file + ".new"))

    meta = {"snapshot": str(snap.name), "last_step_old": int(last_step_old)}
    (run / _MARK).write_text(json.dumps(meta, indent=2))
    print(f"[extend] Snapshot '{snap.name}' created in {run}. last_step_old={last_step_old}")


def merge_extension(run_dir: str, new_file: str = "simfile.dat", out_file: str = "simfile.dat"):
    """
    Merge new simfile into snapshot, renumbering the new block to be contiguous.
    """
    run = Path(run_dir)
    meta_p = run / _MARK
    if not meta_p.exists():
        raise RuntimeError(f"No extension metadata found in {run}. Run start_extension() first.")

    meta = json.loads(meta_p.read_text())
    snap = run / meta["snapshot"]
    last_step_old = int(meta["last_step_old"])

    hdr_old, dat_old = _read_sim_lines(snap)
    hdr_new, dat_new = _read_sim_lines(run / new_file)

    if not dat_new:
        if snap.exists():
            (run / out_file).write_text(snap.read_text())
        print(f"[extend] No new data found in {new_file}; restored snapshot to {out_file}.")
        return

    steps_new = [s for s, _, _ in dat_new]
    delta = _median_step_interval(steps_new)
    if delta <= 0:
        steps_old = [s for s, _, _ in dat_old]
        delta = _median_step_interval(steps_old) or 1

    renumbered = []
    target_step = last_step_old + delta
    for (_s, _t, line) in dat_new:
        renumbered.append(_replace_step(line, target_step))
        target_step += delta

    out = run / out_file
    if out.exists():
        (run / (out_file + ".backup")).write_text(out.read_text())

    headers = hdr_old.copy()
    if hdr_new and hdr_new != hdr_old:
        headers += ["# --- appended after extension ---\n"]

    with out.open("w") as f:
        for h in headers:
            f.write(h)
        for _s, _t, line in dat_old:
            f.write(line if line.endswith("\n") else line + "\n")
        for line in renumbered:
            f.write(line)

    print(f"[extend] Merged into {out.name}. Old last step={last_step_old}, new last step={target_step - delta}.")


def renumber_simfile_in_place(path: str):
    """
    Rewrite step IDs to a contiguous sequence with the median Δstep of the file.
    """
    p = Path(path)
    headers, data = _read_sim_lines(p)
    steps = [s for s, _, _ in data]
    if not steps:
        return
    delta = _median_step_interval(steps) or 1
    first = steps[0]
    with p.open("w") as f:
        for h in headers:
            f.write(h)
        s = first
        for _, __, line in data:
            f.write(_replace_step(line, s))
            s += delta


# ======================================
# Calc-wide convenience (optional)
# ======================================
def _iter_all_run_dirs(calc):
    for leg in calc.legs:
        for stage in leg.stages:
            for lam_window in stage.lam_windows:
                for sim in lam_window.sims:
                    yield sim.output_dir


def prepare_all_runs_for_extension(calc, snapshot_name="simfile.preextend.dat"):
    logger = calc._logger
    n = 0
    for run_dir in _iter_all_run_dirs(calc):
        try:
            start_extension(run_dir, old_file="simfile.dat", snapshot_name=snapshot_name)
            n += 1
        except Exception as e:
            logger.warning(f"[extend] Skipped {run_dir}: {e}")
    logger.info(f"[extend] Prepared {n} run directories for extension.")


def merge_all_extensions(calc, new_file="simfile.dat", out_file="simfile.dat"):
    logger = calc._logger
    ok = 0
    for run_dir in _iter_all_run_dirs(calc):
        try:
            merge_extension(run_dir, new_file=new_file, out_file=out_file)
            ok += 1
        except Exception as e:
            logger.warning(f"[extend] Merge skipped for {run_dir}: {e}")
    logger.info(f"[extend] Completed merge in {ok} run directories.")


# ======================================
# Lambda-specific convenience
# ======================================
# ---------- call this BEFORE you relaunch SOMD for a given λ ----------
def start_extension_for_lambda(calc, leg, stage, lam):
    """
    For the specified λ-window:
      - Move simfile.dat → simfile.preext.<timestamp>.dat so SOMD writes a fresh file.
      - Write extend_meta.json with last_step for later merge.

    Returns True if all runs were prepared (ok even if some runs had no simfile.dat).
    """
    logger = calc._logger
    found = _find_window(calc, leg, stage, lam)
    if not found:
        logger.error(f"[extend] λ-window not found: leg={leg}, stage={stage}, λ={lam}")
        return False

    _, _, win = found
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ok = True

    for sim in win.sims:
        simdir = sim.output_dir
        simfile = os.path.join(simdir, "simfile.dat")

        last_step = None
        pre_snapshot = None

        if os.path.exists(simfile) and os.path.getsize(simfile) > 0:
            with open(simfile, "r") as f:
                lines = f.readlines()
            _, data = _split_header_and_data(lines)
            steps = [st for st in (_parse_step_from_line(ln) for ln in data) if st is not None]
            last_step = steps[-1] if steps else None

            # move aside so SOMD writes a fresh file
            pre_snapshot = os.path.join(simdir, f"simfile.preext.{ts}.dat")
            shutil.move(simfile, pre_snapshot)
            logger.info(f"[extend] {simdir}: moved simfile.dat → {os.path.basename(pre_snapshot)} "
                        f"(last_step={last_step})")
        else:
            logger.warning(f"[extend] {simdir}: no existing simfile.dat; nothing to snapshot")

        meta = {
            "snapshot": os.path.basename(pre_snapshot) if pre_snapshot else None,
            "last_step_before_extension": int(last_step) if last_step is not None else None,
            "created_at": ts,
            "note": "simfile moved aside so SOMD writes a fresh file"
        }
        with open(os.path.join(simdir, "extend_meta.json"), "w") as jf:
            json.dump(meta, jf, indent=2)

        # handy marker to see fresh writes happened
        open(os.path.join(simdir, "simfile.FRESH_PENDING"), "w").close()

    return ok


# ---------- call this AFTER the extension run finished for that λ ----------
def merge_extension_for_lambda(calc, leg, stage, lam, *, renumber_post=True):
    """
    Merge a λ window’s fresh simfile.dat into the pre-extension snapshot for each run:
      - Appends only lines strictly after last_step_before_extension.
      - If renumber_post=True, shifts post steps to be contiguous (no gap).
      - Writes simfile.dat.bak and replaces atomically.

    Returns True if all runs merged successfully.
    """
    logger = calc._logger
    found = _find_window(calc, leg, stage, lam)
    if not found:
        logger.error(f"[extend] λ-window not found: leg={leg}, stage={stage}, λ={lam}")
        return False

    _, _, win = found
    all_ok = True

    for sim in win.sims:
        simdir   = sim.output_dir
        simfile  = os.path.join(simdir, "simfile.dat")
        meta_path= os.path.join(simdir, "extend_meta.json")

        if not (os.path.exists(meta_path) and os.path.exists(simfile)):
            logger.warning(f"[extend] {simdir}: missing extend_meta.json or simfile.dat; skipping")
            all_ok = False
            continue

        with open(meta_path, "r") as jf:
            meta = json.load(jf)

        snap_name = meta.get("snapshot")
        last_keep = meta.get("last_step_before_extension")
        snap_path = os.path.join(simdir, snap_name) if snap_name else None

        if not snap_path or not os.path.exists(snap_path):
            logger.warning(f"[extend] {simdir}: snapshot not found; cannot merge safely")
            all_ok = False
            continue

        # read snapshot and fresh file
        with open(snap_path, "r") as f:
            snap_lines = f.readlines()
        with open(simfile, "r") as f:
            fresh_lines = f.readlines()

        snap_header,  snap_data  = _split_header_and_data(snap_lines)
        fresh_header, fresh_data = _split_header_and_data(fresh_lines)
        header = snap_header if snap_header else fresh_header

        # pre block steps and median interval (fallback = 1000)
        pre_steps = [st for st in (_parse_step_from_line(ln) for ln in snap_data) if st is not None]
        pre_last  = pre_steps[-1] if pre_steps else None
        if len(pre_steps) >= 3:
            median_interval = int(np.median(np.diff(pre_steps)))
            if median_interval <= 0:
                median_interval = 1
        else:
            median_interval = 1000

        # fresh lines strictly after last_keep
        post_raw, post_steps = [], []
        for ln in fresh_data:
            st = _parse_step_from_line(ln)
            if st is None:
                continue
            if (last_keep is None) or (st > int(last_keep)):
                post_raw.append(ln)
                post_steps.append(st)

        if not post_raw:
            logger.info(f"[extend] {simdir}: no new lines to append; leaving as-is")
            try:
                os.remove(os.path.join(simdir, "simfile.FRESH_PENDING"))
            except FileNotFoundError:
                pass
            continue

        # renumber to remove the gap (optional)
        offset_applied = 0
        if renumber_post and pre_last is not None:
            first_post   = post_steps[0]
            target_first = pre_last + median_interval
            offset_applied = first_post - target_first  # subtract from post steps

            renumbered = []
            for ln, st in zip(post_raw, post_steps):
                new_step = int(st - offset_applied)
                if new_step <= pre_last:  # keep strictly increasing
                    new_step = pre_last + median_interval
                rest = ln.strip().split(maxsplit=1)
                new_ln = f"{new_step}\n" if len(rest) == 1 else f"{new_step} {rest[1]}\n"
                renumbered.append(new_ln)

            logger.info(
                f"[extend] {simdir}: de-gapped post block (shift -{offset_applied}, "
                f"interval={median_interval})"
            )
        else:
            renumbered = [ln if ln.endswith('\n') else ln + '\n' for ln in post_raw]
            if not renumber_post:
                logger.info(f"[extend] {simdir}: kept original post steps (gap retained)")

        # merge (drop overlaps)
        merged = list(snap_data)
        for ln in renumbered:
            st = _parse_step_from_line(ln)
            if st is None or (pre_last is not None and st <= pre_last):
                continue
            merged.append(ln)

        # safe write
        bak = simfile + ".bak"
        out = simfile + ".merged"
        shutil.copy2(simfile, bak)
        with open(out, "w") as f:
            for ln in header:
                f.write(ln if ln.endswith("\n") else ln + "\n")
            for ln in merged:
                f.write(ln if ln.endswith("\n") else ln + "\n")
        os.replace(out, simfile)

        # update meta (optional provenance)
        meta.setdefault("merge", {})
        meta["merge"]["renumber_post"] = bool(renumber_post)
        meta["merge"]["median_interval"] = int(median_interval)
        meta["merge"]["offset_applied"] = int(offset_applied)
        meta["merge"]["pre_last_step"] = int(pre_last) if pre_last is not None else None
        meta["merge"]["first_post_raw_step"] = int(post_steps[0])
        with open(meta_path, "w") as jf:
            json.dump(meta, jf, indent=2)

        # clean marker
        try:
            os.remove(os.path.join(simdir, "simfile.FRESH_PENDING"))
        except FileNotFoundError:
            pass

        appended = len(merged) - len(snap_data)
        logger.info(f"[extend] {simdir}: merged; appended {appended} lines; backup -> {bak}")

    return all_ok