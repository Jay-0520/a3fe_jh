#!/usr/bin/env python3
"""
Simplified simulation time fixing utility.

This script identifies problematic λ-windows (gaps, inconsistencies)
and handles them by backing up the current state and preparing for a clean restart.

Key principles:
1. Detect issues systematically
2. Backup problematic runs 
3. Clean restart problematic windows
4. Verify results

Author: Cleaned up from original fix_simulation_times.py
"""

import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging
import numpy as np


def setup_logging(calc_dir: str) -> logging.Logger:
    log_file = Path(calc_dir) / f"simulation_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # force=True ensures we reconfigure even if something configured logging earlier
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # <-- important: allow INFO to pass
    logger.propagate = True        # (default True; keep it explicit)

    logger.info("Starting simulation cleanup process")
    logger.info(f"Log file: {log_file}")
    return logger


def find_window(calc, leg_name: str, stage_name: str, lam: float):
    """Find a specific λ-window in the calculation."""
    leg = next((L for L in calc.legs if L.leg_type.name.lower() == str(leg_name).lower()), None)
    if not leg:
        return None
    stage = next((S for S in leg.stages if S.stage_type.name.lower() == str(stage_name).lower()), None)
    if not stage:
        return None
    win = next((W for W in stage.lam_windows if abs(W.lam - float(lam)) < 1e-9), None)
    return leg, stage, win if win else None


def get_simfile_end_time(sim, timestep_ns: float = 4e-6, detect_gaps: bool = True) -> Tuple[float, bool, Dict]:
    """
    Get the end time of the first contiguous block in a simulation file.
    
    Returns:
        end_time_ns: End time of first contiguous block
        had_gap: Whether gaps were detected
        details: Additional information about the analysis
    """
    simfile_path = Path(sim.output_dir) / "simfile.dat"
    
    if not simfile_path.exists() or simfile_path.stat().st_size == 0:
        return 0.0, False, {"status": "no_file"}
    
    steps = []
    with open(simfile_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            try:
                step = int(line.split()[0])
                steps.append(step)
            except (ValueError, IndexError):
                continue
    
    if not steps:
        return 0.0, False, {"status": "no_data"}
    
    if len(steps) < 2 or not detect_gaps:
        return steps[-1] * timestep_ns, False, {
            "status": "complete",
            "start_step": steps[0],
            "end_step": steps[-1],
            "total_steps": len(steps)
        }
    
    # Detect gaps
    intervals = np.diff(steps)
    median_interval = np.median(intervals) if len(intervals) else 0
    
    if median_interval <= 0:
        return steps[-1] * timestep_ns, False, {
            "status": "complete",
            "start_step": steps[0], 
            "end_step": steps[-1],
            "total_steps": len(steps)
        }
    
    # Look for gaps larger than 2x median interval
    gap_threshold = 2 * median_interval
    large_gaps = np.where(intervals > gap_threshold)[0]
    
    if len(large_gaps) == 0:
        return steps[-1] * timestep_ns, False, {
            "status": "complete",
            "start_step": steps[0],
            "end_step": steps[-1], 
            "total_steps": len(steps)
        }
    
    # Gap detected - return end of first contiguous block
    end_idx = large_gaps[0]
    return steps[end_idx] * timestep_ns, True, {
        "status": "gap_detected",
        "start_step": steps[0],
        "end_step": steps[end_idx],
        "first_block_steps": end_idx + 1,
        "total_steps": len(steps),
        "gap_at_step": steps[end_idx + 1],
        "median_interval": median_interval
    }


def analyze_window_times(calc, logger: logging.Logger) -> Dict[Tuple[str, str, float], Dict]:
    """
    Analyze simulation times for all λ-windows.
    
    Returns:
        Dictionary mapping (leg, stage, lambda) to analysis results
    """
    results = {}
    
    for leg in calc.legs:
        leg_name = leg.leg_type.name
        for stage in leg.stages:
            stage_name = stage.stage_type.name
            for win in stage.lam_windows:
                lam = float(win.lam)
                key = (leg_name, stage_name, lam)
                
                sim_times = []
                had_gaps = []
                details_list = []
                
                for sim in win.sims:
                    time_ns, gap, details = get_simfile_end_time(sim)
                    sim_times.append(time_ns)
                    had_gaps.append(gap)
                    details_list.append(details)
                
                if sim_times:
                    times_array = np.array(sim_times)
                    min_time = np.min(times_array)
                    max_time = np.max(times_array)
                    range_time = max_time - min_time
                    median_time = np.median(times_array)
                    
                    results[key] = {
                        "times": sim_times,
                        "had_gaps": had_gaps,
                        "details": details_list,
                        "min_time": min_time,
                        "max_time": max_time,
                        "range_time": range_time,
                        "median_time": median_time,
                        "n_sims": len(sim_times)
                    }
                    
                    # Log summary
                    gap_count = sum(had_gaps)
                    if gap_count > 0 or range_time > 0.01:  # 0.01 ns threshold
                        logger.warning(f"{leg_name}/{stage_name} λ={lam:.3f}: "
                                     f"times={np.round(sim_times, 6)} ns, "
                                     f"range={range_time:.4f} ns, "
                                     f"gaps={gap_count}/{len(sim_times)}")
                    else:
                        logger.debug(f"{leg_name}/{stage_name} λ={lam:.3f}: ✅ consistent at {min_time:.6f} ns")
    
    return results


def identify_problematic_windows(analysis_results: Dict, 
                                inconsistency_threshold: float = 0.01,
                                logger: Optional[logging.Logger] = None) -> List[Tuple]:
    """
    Identify λ-windows that need restart based on analysis results.
    
    Returns:
        List of (leg, stage, lambda, reason_dict) tuples
    """
    problematic = []
    
    for (leg, stage, lam), results in analysis_results.items():
        reasons = {}
        
        # Check for gaps
        if any(results["had_gaps"]):
            reasons["gaps"] = {
                "count": sum(results["had_gaps"]),
                "total": len(results["had_gaps"])
            }
        
        # Check for time inconsistencies
        if results["range_time"] > inconsistency_threshold:
            reasons["inconsistent_times"] = {
                "range_ns": results["range_time"],
                "threshold_ns": inconsistency_threshold,
                "times": results["times"]
            }
        
        
        if reasons:
            problematic.append((leg, stage, lam, reasons))
            if logger:
                reason_summary = list(reasons.keys())
                logger.info(f"Problematic: {leg}/{stage} λ={lam:.3f} - {reason_summary}")
    
    
    return problematic


def backup_and_clean_window(calc, leg: str, stage: str, lam: float, 
                           backup_dir: Path, logger: logging.Logger,
                           dry_run: bool = False) -> bool:
    """
    Backup a problematic λ-window and clean it for restart.
    
    Returns:
        True if successful, False if failed
    """
    found = find_window(calc, leg, stage, lam)
    if not found or len(found) != 3:
        logger.error(f"λ-window not found: {leg}/{stage} λ={lam}")
        return False
    
    _, _, win = found
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    success = True
    for i, sim in enumerate(win.sims):
        sim_dir = Path(sim.output_dir)
        
        # Create backup directory structure
        rel_path = sim_dir.relative_to(Path(calc.base_dir))
        backup_sim_dir = backup_dir / f"backup_{timestamp}" / rel_path
        
        if dry_run:
            logger.info(f"[DRY RUN] Would backup {sim_dir} -> {backup_sim_dir}")
            logger.info(f"[DRY RUN] Would clean restart files in {sim_dir}")
            continue
        
        try:
            # Create backup
            backup_sim_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup key files
            files_to_backup = [
                "simfile.dat",
                "gradients.dat", 
                "*.s3",
                "*.s3.previous",
                "*.log",
                "*.out",
                "*.err"
            ]
            
            backed_up_files = []
            for pattern in files_to_backup:
                for file_path in sim_dir.glob(pattern):
                    if file_path.is_file():
                        backup_file = backup_sim_dir / file_path.name
                        shutil.copy2(file_path, backup_file)
                        backed_up_files.append(file_path.name)
            
            logger.info(f"Backed up {len(backed_up_files)} files from {sim_dir.name} to {backup_sim_dir}")
            
            # Clean restart files (but keep input files)
            files_to_remove = [
                "simfile.dat",
                "gradients.dat",
                "*.s3", 
                "*.s3.previous"
            ]
            
            removed_files = []
            for pattern in files_to_remove:
                for file_path in sim_dir.glob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
                        removed_files.append(file_path.name)
            
            logger.info(f"Removed {len(removed_files)} restart files from {sim_dir.name}")
            
        except Exception as e:
            logger.error(f"Failed to backup/clean {sim_dir}: {e}")
            success = False
    
    return success


def create_restart_summary(problematic_windows: List[Tuple], 
                          backup_dir: Path, 
                          logger: logging.Logger) -> None:
    """Create a summary file of what was cleaned and needs restart."""
    
    summary_file = backup_dir / "restart_summary.txt"
    
    with open(summary_file, "w") as f:
        f.write(f"Simulation Cleanup Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 50 + "\n\n")
        
        f.write(f"Windows cleaned for restart ({len(problematic_windows)} total):\n\n")
        
        for leg, stage, lam, reasons in problematic_windows:
            f.write(f"• {leg}/{stage} λ={lam:.3f}\n")
            f.write(f"  Reasons: {list(reasons.keys())}\n")
            
            if "gaps" in reasons:
                gap_info = reasons["gaps"]
                f.write(f"  - Gaps detected: {gap_info['count']}/{gap_info['total']} simulations\n")
            
            if "inconsistent_times" in reasons:
                time_info = reasons["inconsistent_times"]
                f.write(f"  - Time range: {time_info['range_ns']:.4f} ns (threshold: {time_info['threshold_ns']:.4f} ns)\n")
            
            if "short_outliers" in reasons:
                outlier_info = reasons["short_outliers"]
                f.write(f"  - Short outliers: {outlier_info['short_count']} simulations below {outlier_info['threshold_fraction']:.0%} of median\n")
            
            f.write("\n")
        
        f.write(f"\nNext steps:\n")
        f.write(f"1. Review this summary\n")
        f.write(f"2. Re-run simulations for the cleaned windows\n") 
        f.write(f"3. Run verification after completion\n")
        f.write(f"\nBackup location: {backup_dir}\n")
    
    logger.info(f"Restart summary written to: {summary_file}")


def main_cleanup_workflow(calc, 
                         inconsistency_threshold: float = 0.01,
                         dry_run: bool = True) -> Dict:
    """
    Main workflow for cleaning up problematic simulations.
    
    Args:
        calc: The calculation object
        inconsistency_threshold: Maximum allowed time range between simulations (ns)
        dry_run: If True, only show what would be done without making changes
        
    Returns:
        Dictionary with cleanup results
    """
    # Setup
    calc_dir = Path(calc.base_dir)
    logger = setup_logging(calc_dir)
    
    backup_base_dir = calc_dir / "cleanup_backups" 
    backup_base_dir.mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("SIMULATION CLEANUP WORKFLOW")
    logger.info("=" * 60)
    logger.info(f"Calculation directory: {calc_dir}")
    logger.info(f"Inconsistency threshold: {inconsistency_threshold:.4f} ns")
    logger.info(f"Dry run mode: {dry_run}")
    
    # Step 1: Analyze all windows
    logger.info("\n1. ANALYZING SIMULATION TIMES...")
    analysis_results = analyze_window_times(calc, logger)
    
    total_windows = len(analysis_results)
    logger.info(f"Analyzed {total_windows} λ-windows")
    
    # Step 2: Identify problematic windows  
    logger.info("\n2. IDENTIFYING PROBLEMATIC WINDOWS...")
    problematic_windows = identify_problematic_windows(
        analysis_results, 
        inconsistency_threshold=inconsistency_threshold,
        logger=logger
    )
    
    if not problematic_windows:
        logger.info("✅ No problematic windows found! All simulations look good.")
        return {
            "success": True,
            "total_windows": total_windows,
            "problematic_windows": 0,
            "cleaned_windows": 0,
            "message": "All simulations are consistent"
        }
    
    logger.info(f"Found {len(problematic_windows)} problematic windows")
    
    # Step 3: Backup and clean
    logger.info(f"\n3. BACKUP AND CLEANUP...")
    
    if dry_run:
        logger.info("DRY RUN MODE - No actual changes will be made")
    
    cleaned_count = 0
    failed_count = 0
    
    for leg, stage, lam, reasons in problematic_windows:
        logger.info(f"Processing {leg}/{stage} λ={lam:.3f}...")
        
        success = backup_and_clean_window(
            calc, leg, stage, lam, backup_base_dir, logger, dry_run=dry_run
        )
        
        if success:
            cleaned_count += 1
        else:
            failed_count += 1
    
    # Step 4: Create summary
    if not dry_run:
        logger.info(f"\n4. CREATING RESTART SUMMARY...")
        create_restart_summary(problematic_windows, backup_base_dir, logger)
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("CLEANUP COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total windows analyzed: {total_windows}")
    logger.info(f"Problematic windows found: {len(problematic_windows)}")
    logger.info(f"Successfully cleaned: {cleaned_count}")
    logger.info(f"Failed to clean: {failed_count}")
    
    if not dry_run and cleaned_count > 0:
        logger.info(f"\n⚠️  IMPORTANT: {cleaned_count} windows have been cleaned and need to be restarted")
        logger.info(f"📁 Backup location: {backup_base_dir}")
        logger.info(f"📋 Review restart summary: {backup_base_dir}/restart_summary.txt")
    
    success = failed_count == 0
    
    return {
        "success": success,
        "total_windows": total_windows,
        "problematic_windows": len(problematic_windows), 
        "cleaned_windows": cleaned_count,
        "failed_windows": failed_count,
        "backup_location": str(backup_base_dir) if not dry_run else None,
        "dry_run": dry_run
    }


# Convenience functions for external use
def fix_simulation_times(calc, **kwargs):
    """
    Simplified interface to the cleanup workflow.
    
    This replaces the complex fix_simulation_times function from the original script
    with a clean, backup-focused approach.
    """
    return main_cleanup_workflow(calc, **kwargs)


def quick_analysis(calc):
    """Quick analysis of simulation times without making changes."""
    return main_cleanup_workflow(calc, dry_run=True)

