import os
import json
import subprocess
import numpy as np
from pathlib import Path
from typing import Tuple, List

# ==================================================
# RUNTIME UTILITIES
# ==================================================
def _replace_step(line: str, new_step: int) -> str:
    """Replace only the first column with new_step, preserve spacing and trailing spaces."""
    idx = line.find(" ")
    if idx == -1:  # defensive, shouldn't happen for valid lines
        return str(new_step) + "\n"
    return f"{new_step}{line[idx:]}"
    
def _get_actual_simtime_from_file(sim, timestep_ns=4e-6, detect_gaps=True):
    """
    Robustly determine the actual simulation time from simfile.dat
    by scanning to the last valid data line, with optional gap detection.
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
                parts = line.split()
                if parts:
                    step = int(parts[0])
                    steps.append(step)
            except ValueError:
                continue

    if not steps:
        return 0.0
    
    if not detect_gaps:
        return steps[-1] * timestep_ns
    
    # Detect gaps in step sequence
    if len(steps) < 2:
        return steps[-1] * timestep_ns
    
    # Calculate step intervals
    intervals = np.diff(steps)
    median_interval = np.median(intervals) if len(intervals) else 0
    
    # Find large gaps (more than 2x the median interval)
    gap_threshold = 2 * median_interval if median_interval > 0 else 0
    large_gaps = np.where(intervals > gap_threshold)[0] if gap_threshold > 0 else []

    if len(large_gaps) == 0:
        return steps[-1] * timestep_ns
    
    # Always use the first contiguous block (from the beginning)
    first_gap_idx = large_gaps[0]  # Index of first large gap
    start_idx = 0
    end_idx = first_gap_idx  # End at the first gap (inclusive)
    block_size = end_idx - start_idx + 1
    
    if detect_gaps:
        sim._logger.info(f"Gap detection results for {simfile_path}:")
        sim._logger.info(f"  Total steps: {len(steps)}")
        sim._logger.info(f"  Median interval: {median_interval}")
        sim._logger.info(f"  Large gaps found: {len(large_gaps)}")
        sim._logger.info(f"  [ACTUAL SIMTIME] Using first contiguous block: steps {steps[start_step:=start_idx]} to {steps[end_idx]} ({block_size} entries)")
        
    # Return time based on largest contiguous block
    return steps[end_idx] * timestep_ns


def _get_first_block_end_time(sim, timestep_ns=None, gap_factor=2.0):
    """
    Compute the end time (ns) of the first contiguous block.
    If no gap is found, returns the total time and had_gap=False.
    """
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
            except ValueError:
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

    end_idx = large_gaps[0]  # last sample before the gap
    details = {"start_step": steps[0], "end_step": steps[end_idx], "block_len": end_idx + 1}
    return steps[end_idx] * timestep_ns, True, details


def truncate_simulations_to_minimum(calc, detect_gaps=True):
    """
    Truncate all simulations to the minimum runtime for each lambda window,
    using robust parsing of simfile.dat instead of get_tot_simtime().

    UPDATED POLICY: if a gap exists in any sim, ALWAYS truncate that sim
    to the end of its first contiguous block (delete the rest), regardless
    of cross-run consistency.
    """
    logger = calc._logger

    for leg in calc.legs:
        for stage in leg.stages:
            stage_has_issues = False

            for lam_window in stage.lam_windows:
                sim_times = []
                had_gaps = []       
                first_block_times = [] 
                fb_details = []        

                for sim in lam_window.sims:
                    # Always compute first-block end time
                    fb_time, had_gap, details = _get_first_block_end_time(
                        sim, timestep_ns=getattr(sim, "timestep", 4e-6), gap_factor=2.0
                    )  
                    first_block_times.append(fb_time)  
                    had_gaps.append(had_gap)         
                    fb_details.append(details)       

                    # For consistency checks, use the "effective" time we intend to rely on
                    sim_times.append(fb_time) 

                    if had_gap:
                        logger.warning(
                            f"  Gap detected in {os.path.join(sim.output_dir, 'simfile.dat')}: "
                            f"keeping first block steps {details.get('start_step')}–{details.get('end_step')} "
                            f"({details.get('block_len')} entries)"
                        )

                # If any sim has a gap, enforce the first-block trimming per sim  
                if detect_gaps and any(had_gaps): 
                    logger.info(
                        f"❌ Leg {leg.leg_type.name} Stage {stage.stage_type.name} Lambda {lam_window.lam:.3f}: gaps detected → trimming each run to its first contiguous block"
                    )
                    for sim, fb_time in zip(lam_window.sims, first_block_times):
                        truncate_simulation_file(sim, fb_time, logger, detect_gaps=False)  # (turn off inner gap logic)
                    # After explicit trimming, no need for min-time truncation on this λ-window  
                    continue  

                # No gaps at all → fall back to min-time truncation if inconsistent (old behavior)
                min_time = min(sim_times)
                max_time = max(sim_times)

                if abs(max_time - min_time) > 0.01:
                    stage_has_issues = True
                    logger.warning(f"❌ Leg {leg.leg_type.name} Stage {stage.stage_type.name} Lambda {lam_window.lam:.3f}: Inconsistent times {sim_times}")
                    logger.warning(f"  -> Truncating to minimum: {min_time:.6f} ns")

                    for i, sim in enumerate(lam_window.sims):
                        if sim_times[i] > min_time:
                            truncate_simulation_file(sim, min_time, logger, detect_gaps=False)
                            logger.info(
                                f"     Truncated run {sim.run_no}: {sim_times[i]:.6f} -> {min_time:.6f} ns"
                            )
                else:
                    logger.debug(
                        f"Lambda {lam_window.lam:.3f}: ✅ All runs consistent at {min_time:.6f} ns"
                    )

            if not stage_has_issues:
                logger.info(f" - Leg {leg.leg_type.name} Stage {stage.stage_type.name} has no timing issues")


def truncate_simulation_file(simulation, target_time_ns, logger, detect_gaps=True):
    """
    Truncate a simulation file to a specific time (ns).
    When detect_gaps is True, this function will *also* reduce to the first
    contiguous block *before* applying time-based truncation; however, when
    you already pass target_time_ns as the first-block end, call with
    detect_gaps=False to avoid double work.
    """
    simfile_path = os.path.join(simulation.output_dir, "simfile.dat")

    if not os.path.exists(simfile_path):
        logger.warning(f"Warning: {simfile_path} does not exist, skipping truncation")
        return

    timestep_ns = getattr(simulation, "timestep", 4e-6)

    with open(simfile_path, 'r') as f:
        lines = f.readlines()

    header_lines = []
    steps = []

    for line in lines:
        if line.startswith('#'):
            header_lines.append(line)
        elif line.strip():
            try:
                parts = line.split()
                if parts:
                    step = int(parts[0])
                    time_ns = step * timestep_ns
                    steps.append((step, time_ns, line))
            except (ValueError, IndexError):
                continue

    if not steps:
        logger.warning(f"Warning: No valid data found in {simfile_path}")
        return

    # Optional inner gap handling
    if detect_gaps and len(steps) > 1:
        step_nums = [s[0] for s in steps]
        intervals = np.diff(step_nums)
        median_interval = np.median(intervals) if len(intervals) else 0
        if median_interval > 0: 
            gap_threshold = 2 * median_interval
            large_gaps = np.where(intervals > gap_threshold)[0]

            if len(large_gaps) > 0:
                first_gap_idx = large_gaps[0]
                start_idx = 0
                end_idx = first_gap_idx
                logger.warning(f"     Gap detected in {simfile_path}")
                logger.info(
                    f"     Using first contiguous block: steps {step_nums[start_idx]} to {step_nums[end_idx]} "
                    f"({end_idx - start_idx + 1} entries)"
                )
                steps = steps[start_idx:end_idx + 1]

    # Time-based truncation to target_time_ns
    final_data_lines = []
    for step, time_ns, line in steps:
        if time_ns <= target_time_ns + 1e-9:
            final_data_lines.append(line)
        else:
            break

    if not final_data_lines:
        logger.warning(f"Warning: No valid data found within target time in {simfile_path}")
        return

    # Backup
    backup_path = simfile_path + ".backup"
    if not os.path.exists(backup_path):
        subprocess.run(['cp', simfile_path, backup_path], check=False)

    # Write truncated file
    with open(simfile_path, 'w') as f:
        for line in header_lines:
            f.write(line)
        for line in final_data_lines:
            f.write(line)

    last_step = int(final_data_lines[-1].split()[0])
    actual_time = last_step * timestep_ns
    logger.info(f"     Truncated to step {last_step}, actual time: {actual_time:.6f} ns")


def verify_truncation(calc):
    """
    Verify using the *post-trim* definition of time:
    simply read the last kept step (no gap logic), which reflects the file on disk.
    """
    logger = calc._logger
    all_consistent = True

    for leg in calc.legs:
        for stage in leg.stages:
            for lam_window in stage.lam_windows:
                sim_times = []
                for sim in lam_window.sims:
                    # Use raw last-step time from file (no gap analysis)
                    t_ns = _get_actual_simtime_from_file(
                        sim,
                        timestep_ns=getattr(sim, "timestep", 4e-6),
                        detect_gaps=True, 
                    )
                    sim_times.append(t_ns)

                min_time = min(sim_times) if sim_times else 0.0
                max_time = max(sim_times) if sim_times else 0.0

                if abs(max_time - min_time) > 0.01:
                    logger.error(
                        f"Leg {leg.leg_type.name} Stage {stage.stage_type.name} "
                        f"Lambda {lam_window.lam:.3f}: ❌ Still inconsistent: {sim_times}"
                    )
                    all_consistent = False
                else:
                    logger.debug(
                        f"Leg {leg.leg_type.name} Stage {stage.stage_type.name} "
                        f"Lambda {lam_window.lam:.3f}: ✅ Consistent at {min_time:.6f} ns"
                    )

    if all_consistent:
        logger.info("✅ ALL SIMULATIONS NOW HAVE CONSISTENT TIMES!")
    else:
        logger.error("❌ Some simulations still have inconsistent times")

    return all_consistent


def fix_simulation_times(calc, apply_truncation=True, detect_gaps=True):
    """
    Complete workflow to fix inconsistent simulation times.
    """
    logger = calc._logger
    if apply_truncation:
        logger.info("Starting simulation time fixing process...")
        truncate_simulations_to_minimum(calc, detect_gaps=detect_gaps)

    success = verify_truncation(calc)
    
    if success:
        logger.info("\n🎉 Ready to proceed with next steps!")
    else:
        logger.warning("\n⚠️  Some issues remain. Manual inspection may be required.")
    
    return success


# ============================================
# EXTENSION-SAFE RESUME / MERGE UTILITIES
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
    Prepare to extend a run by snapshotting the current simfile and recording last_step_old.
    After calling this, run SOMD again (ideally writing a fresh 'simfile.dat').
    """
    run = Path(run_dir)
    old = run / old_file
    snap = run / snapshot_name

    headers, data = _read_sim_lines(old)
    last_step_old = data[-1][0] if data else 0

    # Make a snapshot of the current file
    if old.exists():
        snap.write_text(old.read_text())
        # Move the old file away so SOMD writes a fresh one (optional but recommended)
        old.rename(run / (old_file + ".new"))

    # Record meta
    meta = {"snapshot": str(snap.name), "last_step_old": int(last_step_old)}
    (run / _MARK).write_text(json.dumps(meta, indent=2))
    print(f"[extend] Snapshot '{snap.name}' created in {run}. last_step_old={last_step_old}")

def merge_extension(run_dir: str, new_file: str = "simfile.dat", out_file: str = "simfile.dat"):
    """
    Merge newly produced file into the snapshot, renumbering the new block
    to be contiguous with the snapshot’s last step.
    """
    run = Path(run_dir)
    meta_p = run / _MARK
    if not meta_p.exists():
        raise RuntimeError(f"No extension metadata found in {run}. Run start_extension() first.")

    meta = json.loads(meta_p.read_text())
    snap = run / meta["snapshot"]
    last_step_old = int(meta["last_step_old"])

    # Read snapshot and new
    hdr_old, dat_old = _read_sim_lines(snap)
    hdr_new, dat_new = _read_sim_lines(run / new_file)

    # If nothing new, just restore snapshot as output
    if not dat_new:
        if snap.exists():
            (run / out_file).write_text(snap.read_text())
        print(f"[extend] No new data found in {new_file}; restored snapshot to {out_file}.")
        return

    steps_new = [s for s, _, _ in dat_new]
    delta = _median_step_interval(steps_new)
    if delta <= 0:
        # fall back to old block delta or 1
        steps_old = [s for s, _, _ in dat_old]
        delta = _median_step_interval(steps_old) or 1

    # Build renumbered lines for the new block
    renumbered = []
    target_step = last_step_old + delta  # first step after snapshot
    for (_s, _t, line) in dat_new:
        renumbered.append(_replace_step(line, target_step))
        target_step += delta

    # Write a backup if the out_file exists
    out = run / out_file
    if out.exists():
        (run / (out_file + ".backup")).write_text(out.read_text())

    # Merge headers: keep old headers; if new has extra headers, add a marker
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
    Full renormalization: rewrite step IDs to a contiguous sequence with the median Δstep of the file.
    Useful when you already have gaps/overlaps and just want a clean, continuous time axis.
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
# CALC-WIDE CONVENIENCE WRAPPERS (A3FE)
# ======================================
def _iter_all_run_dirs(calc):
    """Yield each run directory path (string) for all sims in the calculation."""
    for leg in calc.legs:
        for stage in leg.stages:
            for lam_window in stage.lam_windows:
                for sim in lam_window.sims:
                    yield sim.output_dir  # expected to contain simfile.dat

def prepare_all_runs_for_extension(calc, snapshot_name="simfile.preextend.dat"):
    """
    For every run directory in the calc, snapshot the current simfile and
    move it aside so resumed SOMD writes a fresh file.
    """
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
    """
    For every run directory in the calc, merge the newly produced simfile into
    the snapshot, renumbering appended lines for continuity.
    """
    logger = calc._logger
    ok = 0
    for run_dir in _iter_all_run_dirs(calc):
        try:
            merge_extension(run_dir, new_file=new_file, out_file=out_file)
            ok += 1
        except Exception as e:
            logger.warning(f"[extend] Merge skipped for {run_dir}: {e}")
    logger.info(f"[extend] Completed merge in {ok} run directories.")
