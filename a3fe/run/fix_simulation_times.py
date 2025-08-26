import os
import subprocess
import numpy as np


def _get_actual_simtime_from_file(sim, timestep_ns=4e-6, detect_gaps=True):
    """
    Robustly determine the actual simulation time from simfile.dat
    by scanning to the last valid data line, with optional gap detection.
    
    Parameters
    ----------
    sim : a3fe.Simulation
        Simulation object
    timestep_ns : float
        Simulation timestep in ns (default 4 fs = 4e-6 ns)
    detect_gaps : bool
        If True, detect large gaps in step numbers and truncate to largest contiguous block
    
    Returns
    -------
    float
        Actual usable simulation time in ns
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
    
    sim._logger.info(f"Gap detection results for {simfile_path}:")
    sim._logger.info(f"  Total steps: {len(steps)}")
    sim._logger.info(f"  Median interval: {median_interval}")
    sim._logger.info(f"  Large gaps found: {len(large_gaps)}")
    
    # Return time based on largest contiguous block
    return steps[end_idx] * timestep_ns


def truncate_simulations_to_minimum(calc, detect_gaps=True):
    """
    Truncate all simulations to the minimum runtime for each lambda window,
    using robust parsing of simfile.dat instead of get_tot_simtime().
    
    Parameters
    ----------
    calc : a3fe.Calculation
        The calculation object with potentially inconsistent simulation times
    """
    logger = calc._logger
    logger.info("=== TRUNCATING SIMULATIONS TO MINIMUM RUNTIME ===")
    
    for leg in calc.legs:
        logger.info(f"=== {leg.leg_type.name} LEG ===")
        
        for stage in leg.stages:
            logger.info(f"--- {stage.stage_type.name} STAGE ---")
            stage_has_issues = False
            
            for lam_window in stage.lam_windows:
                # Get simulation times for all runs (robust)
                sim_times = []
                for sim in lam_window.sims:
                    sim_time = _get_actual_simtime_from_file(sim, timestep_ns=sim.timestep, detect_gaps=detect_gaps)
                    sim_times.append(sim_time)
                
                min_time = min(sim_times)
                max_time = max(sim_times)
                
                if abs(max_time - min_time) > 0.01:
                    stage_has_issues = True
                    logger.warning(f"Lambda {lam_window.lam:.3f}: Inconsistent times {sim_times}")
                    logger.warning(f"  -> Truncating to minimum: {min_time:.6f} ns")
                    
                    for i, sim in enumerate(lam_window.sims):
                        if sim_times[i] > min_time:
                            truncate_simulation_file(sim, min_time, logger, detect_gaps=detect_gaps)
                            logger.info(f"     Truncated run {sim.run_no}: {sim_times[i]:.6f} -> {min_time:.6f} ns")
                else:
                    logger.debug(f"Lambda {lam_window.lam:.3f}: ✓ All runs consistent at {min_time:.6f} ns")
            
            if not stage_has_issues:
                logger.info(f"  ✓ Stage {stage.stage_type.name} has no timing issues")
    
    logger.info("=== TRUNCATION COMPLETE ===")


def truncate_simulation_file(simulation, target_time_ns, logger, detect_gaps=True):
    """
    Truncate a simulation file to a specific time.
    
    Parameters
    ----------
    simulation : a3fe.Simulation
        The simulation object to truncate
    target_time_ns : float
        Target simulation time in nanoseconds
    """
    
    simfile_path = os.path.join(simulation.output_dir, "simfile.dat")
    
    if not os.path.exists(simfile_path):
        logger.warning(f"Warning: {simfile_path} does not exist, skipping truncation")
        return
    
    # For SOMD simfiles: each step = 4 fs, so time_ns = steps * 4e-6
    # TODO: hardcoded timestep for now, but it should be configurable 
    timestep_ns = 4e-6  # 4 fs converted to ns
    
    # Read the simulation file
    with open(simfile_path, 'r') as f:
        lines = f.readlines()
    
    # Find data lines (non-comment lines)
    header_lines = []
    steps = []
    
    # First pass: collect all steps and identify structure
    for line in lines:
        if line.startswith('#'):
            header_lines.append(line)
        elif line.strip():  # Non-empty line
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
    
    # Determine which steps to keep
    if detect_gaps and len(steps) > 1:
        # Extract just step numbers for gap analysis
        step_nums = [s[0] for s in steps]
        intervals = np.diff(step_nums)
        median_interval = np.median(intervals)
        gap_threshold = 3 * median_interval
        large_gaps = np.where(intervals > gap_threshold)[0]
        
        if len(large_gaps) > 0:
            # Use the first contiguous block (from beginning to first gap)
            first_gap_idx = large_gaps[0]
            start_idx = 0
            end_idx = first_gap_idx  # End at first gap (inclusive)
            
            logger.info(f"     Gap detected in {simfile_path}")
            logger.info(f"     Using first contiguous block: steps {step_nums[start_idx]} to {step_nums[end_idx]} ({end_idx - start_idx + 1} entries)")
            
            # Filter steps to first contiguous block
            steps = steps[start_idx:end_idx + 1]


    # Now apply time-based truncation to the (possibly gap-filtered) steps
    final_data_lines = []
    for step, time_ns, line in steps:
        if time_ns <= target_time_ns + 1e-9:  # Small tolerance for floating point
            final_data_lines.append(line)
        else:
            break
    
    if not final_data_lines:
        logger.warning(f"Warning: No valid data found within target time in {simfile_path}")
        return
    
    # Create backup
    backup_path = simfile_path + ".backup"
    if not os.path.exists(backup_path):
        subprocess.run(['cp', simfile_path, backup_path])
    
    # Write truncated file
    with open(simfile_path, 'w') as f:
        for line in header_lines:
            f.write(line)
        
        for line in final_data_lines:
            f.write(line)
    
    # Verify the truncation worked
    last_step = int(final_data_lines[-1].split()[0])
    actual_time = last_step * timestep_ns
    
    logger.info(f"     Truncated to step {last_step}, actual time: {actual_time:.6f} ns")


def verify_truncation(calc):
    logger = calc._logger    
    all_consistent = True

    for leg in calc.legs:        
        for stage in leg.stages:            
            for lam_window in stage.lam_windows:
                sim_times = []
                for sim in lam_window.sims:
                    sim_time = sim.get_tot_simtime()
                    sim_times.append(sim_time)
                
                min_time = min(sim_times)
                max_time = max(sim_times)
                
                if abs(max_time - min_time) > 0.01:
                    logger.error(f"Leg {leg.leg_type.name} Stage {stage.stage_type.name} Lambda {lam_window.lam:.3f}: ❌ Still inconsistent: {sim_times}")
                    all_consistent = False
                else:
                    logger.debug(f"Leg {leg.leg_type.name} Stage {stage.stage_type.name} Lambda {lam_window.lam:.3f}: ✓ Consistent at {min_time:.6f} ns")
    
    if all_consistent:
        logger.info("✅ ALL SIMULATIONS NOW HAVE CONSISTENT TIMES!")
    else:
        logger.error("❌ Some simulations still have inconsistent times")
    
    return all_consistent


def fix_simulation_times(calc, apply_truncation=True):
    """
    Complete workflow to fix inconsistent simulation times.
    
    Parameters
    ----------
    calc : a3fe.Calculation
        The calculation object to fix
    """
    logger = calc._logger
    if apply_truncation:
        logger.info("Starting simulation time truncation process...")
        truncate_simulations_to_minimum(calc)

    success = verify_truncation(calc)
    
    if success:
        logger.info("\n🎉 Ready to proceed with next steps!")
    else:
        logger.warning("\n⚠️  Some issues remain. Manual inspection may be required.")
    
    return success

