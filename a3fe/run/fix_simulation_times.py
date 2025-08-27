import os
import subprocess
import numpy as np


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
    median_interval = np.median(intervals)
    
    # Find large gaps (more than 2x the median interval)
    gap_threshold = 2 * median_interval
    large_gaps = np.where(intervals > gap_threshold)[0]
    
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
        sim._logger.info(f"  [ACTUAL SIMTIME] Using first contiguous block: steps {steps[start_idx]} to {steps[end_idx]} ({block_size} entries)")
        
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
    median_interval = np.median(intervals)

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
    logger.info("=== TRUNCATING SIMULATIONS TO MINIMUM RUNTIME ===")

    for leg in calc.legs:
        logger.info(f"=== {leg.leg_type.name} LEG ===")

        for stage in leg.stages:
            logger.info(f"--- {stage.stage_type.name} STAGE ---")
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
                        f"Lambda {lam_window.lam:.3f}: gaps detected → trimming each run to its first contiguous block"  
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
                    logger.warning(f"Lambda {lam_window.lam:.3f}: Inconsistent times {sim_times}")
                    logger.warning(f"  -> Truncating to minimum: {min_time:.6f} ns")

                    for i, sim in enumerate(lam_window.sims):
                        if sim_times[i] > min_time:
                            truncate_simulation_file(sim, min_time, logger, detect_gaps=False)
                            logger.info(
                                f"     Truncated run {sim.run_no}: {sim_times[i]:.6f} -> {min_time:.6f} ns"
                            )
                else:
                    logger.debug(
                        f"Lambda {lam_window.lam:.3f}: ✓ All runs consistent at {min_time:.6f} ns"
                    )

            if not stage_has_issues:
                logger.info(f"  ✓ Stage {stage.stage_type.name} has no timing issues")

    logger.info("=== TRUNCATION COMPLETE ===")


def truncate_simulation_file(simulation, target_time_ns, logger, detect_gaps=True):
    """
    Truncate a simulation file to a specific time (ns).
    When detect_gaps is True, this function will *also* reduce to the first
    contiguous block *before* applying time-based truncation; however, when
    you already pass target_time_ns as the first-block end, call with
    detect_gaps=False to avoid double work.  # UPDATED
    """
    simfile_path = os.path.join(simulation.output_dir, "simfile.dat")

    if not os.path.exists(simfile_path):
        logger.warning(f"Warning: {simfile_path} does not exist, skipping truncation")
        return

    # Use the simulation's timestep if available 
    # TODO: or we should get this from somd.cfg file?
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

    # Optional inner gap handling (can be disabled by caller) 
    if detect_gaps and len(steps) > 1:
        step_nums = [s[0] for s in steps]
        intervals = np.diff(step_nums)
        median_interval = np.median(intervals)
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
        subprocess.run(['cp', simfile_path, backup_path])

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
                        detect_gaps=False, 
                    )
                    sim_times.append(t_ns)

                min_time = min(sim_times)
                max_time = max(sim_times)

                if abs(max_time - min_time) > 0.01:
                    logger.error(
                        f"Leg {leg.leg_type.name} Stage {stage.stage_type.name} "
                        f"Lambda {lam_window.lam:.3f}: ❌ Still inconsistent: {sim_times}"
                    )
                    all_consistent = False
                else:
                    logger.debug(
                        f"Leg {leg.leg_type.name} Stage {stage.stage_type.name} "
                        f"Lambda {lam_window.lam:.3f}: ✓ Consistent at {min_time:.6f} ns"
                    )

    if all_consistent:
        logger.info("✅ ALL SIMULATIONS NOW HAVE CONSISTENT TIMES!")
    else:
        logger.error("❌ Some simulations still have inconsistent times")

    return all_consistent



def fix_simulation_times(calc, apply_truncation=True, detect_gaps=True):
    """
    Complete workflow to fix inconsistent simulation times.
    
    In practice, we might have to run this multiple times if there are
    multiple gaps as well as inconsistent runtimes in different runs.
    """
    logger = calc._logger
    if apply_truncation:
        logger.info("Starting simulation time truncation process...")
        truncate_simulations_to_minimum(calc, detect_gaps=detect_gaps)

    success = verify_truncation(calc)
    
    if success:
        logger.info("\n🎉 Ready to proceed with next steps!")
    else:
        logger.warning("\n⚠️  Some issues remain. Manual inspection may be required.")
    
    return success

