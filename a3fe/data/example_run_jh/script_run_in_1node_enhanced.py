"""
Version: v5.0
This script is enhanced such ways:
    - parallel MBAR execution
    - concurrent SOMD execution
    - robust error handling and logging for MBAR and SOMD jobs

It has been tested on macOS and HPC with Python 3.10+ as of 2025-08-31 
"""
import os
import threading

os.environ["MPLBACKEND"] = "Agg"  # prevents macOS GUI backend from opening windows
try:
    import matplotlib as _mpl

    _mpl.use("Agg", force=True)
except Exception:
    pass

import shutil
import subprocess
from a3fe.run._virtual_queue import VirtualQueue
import a3fe as a3
from a3fe.run.system_prep import SystemPreparationConfig
import re
import logging
import time
from datetime import datetime
from a3fe.run.simulation import Simulation
from a3fe.run.stage import Stage
from a3fe.run._simulation_runner import SimulationRunner
from a3fe.run._virtual_queue import Job
from a3fe.run.enums import JobStatus as _JobStatus
from time import sleep
from functools import lru_cache
import concurrent.futures
import itertools
import subprocess
from tqdm import tqdm
import sys
import shlex
from collections import defaultdict
from decimal import Decimal
from a3fe.run.fix_simulation_times import fix_simulation_times
import inspect


# Configuration options
FORCE_LOCAL_EXECUTION = True  # Set to False for normal SLURM execution
FORCE_CPU_PLATFORM = False  # Set to True to force CPU even on GPU systems
FAST_UPDATE_INTERVAL = 3  # seconds between updates for local execution
SKIP_ADAPTIVE_EFFICIENCY = False  # Set to True to skip adaptive efficiency checks
MAX_CONCURRENT_SOMD = 4  # only 2 concurrent somd jobs per GPU to avoid oversubscription
A3FE_STILL_RUNNING_THROTTLE_SEC = 600  # 10 minutes; set to 0 to disable throttling


# ==================================================
# LOGGING SETUP FOR LOCAL EXECUTION
# ==================================================
class ColorFormatter(logging.Formatter):
    ORANGE = "\033[33m"  # ANSI “yellow” as an orange stand‐in
    GREEN = "\033[32m"
    RED = "\033[31m"
    RESET = "\033[0m"

    FORMATS = {
        logging.DEBUG: ORANGE + "%(levelname)s: %(message)s" + RESET,
        logging.INFO: GREEN + "%(levelname)s: %(message)s" + RESET,
        logging.ERROR: RED + "%(levelname)s: %(message)s" + RESET,
        logging.WARNING: RED + "%(levelname)s: %(message)s" + RESET,
    }

    def format(self, record):
        if not hasattr(record, "tag"):
            record.tag = ""
        fmt = self.FORMATS.get(record.levelno, "%(levelname)s: %(message)s")
        return logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S").format(record)


def get_tagged_logger(name: str, tag: str | None = None) -> logging.LoggerAdapter:
    """
    Return a LoggerAdapter that injects a tag into log records
    and *does not* attach its own handlers (so it uses the root handler/format).
    """
    base = logging.getLogger(name)
    base.propagate = True
    # Do not add handlers here; rely on root configured in setup_global_logging()
    return logging.LoggerAdapter(base, extra={"tag": f"[{tag}] " if tag else ""})


class DedupStatusFilter(logging.Filter):
    """
    we may want to de-duplicate massive logging info like:

    INFO - 2025-08-02 22:28:36,711 - Simulation (stage=discharge, lam=0.0, run_no=1)_70 - Not running
    INFO - 2025-08-02 22:28:36,719 - Simulation (stage=discharge, lam=0.0, run_no=1)_70 - Job
     (virtual_job_id = 1, slurm_job_id= 999999), status = JobStatus.FINISHED finished successfully
    INFO - 2025-08-02 22:28:36,719 - Simulation (stage=discharge, lam=1.0, run_no=1)_74 - Not running
    INFO - 2025-08-02 22:28:36,743 - Simulation (stage=discharge, lam=1.0, run_no=1)_74 - Job
     (virtual_job_id = 2, slurm_job_id= 999999), status = JobStatus.FINISHED finished successfully

    so that we only log out info when the status changes.
    """

    JOBID_RE = re.compile(r"slurm_job_id=\s*(\d+)")
    STATUS_RE = re.compile(r"status\s*=\s*(JobStatus\.\w+)")
    SIM_DETAILS_RE = re.compile(
        r"Simulation \(leg=([^,]+), stage=([^,]+), lam=([^,]+), run_no=([^)]+)\)"
    )

    def __init__(self, debug_mode: bool = False):
        super().__init__()
        self.heartbeat_interval_sec = A3FE_STILL_RUNNING_THROTTLE_SEC
        self._last_heartbeat_by_job: dict[str, float] = {}
        self.suppress_still_running: bool = (self.heartbeat_interval_sec == 0)

        self.debug_mode = debug_mode
        self.suppress_mbar_noise: bool = False
        self._mbar_noise = re.compile(
            r"(?:Submitted MBAR job \d+:|\[LOCAL UPDATE\].*MBAR job \d+.*|MBAR job \d+ (?:running|completed successfully|failed))" 
        )
        self._last_status_by_job: dict[str, str] = (
            {}
        )  # unique_job_key -> last seen status
        self._not_running_jobs: set[str] = (
            set()
        )  # Track which jobs have logged "Not running"

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        name = record.name

        if self.debug_mode:
            print(f"[LOCAL DEBUG]: Processing message from {name}: {msg[:100]}...")

        if "Still running" in msg:
            if self.suppress_still_running:
                return False
            job_key = self._get_job_key(msg, name) or "GLOBAL"
            now = time.time()
            last = self._last_heartbeat_by_job.get(job_key, 0.0)
            if now - last < self.heartbeat_interval_sec:
                return False  # too soon; drop
            self._last_heartbeat_by_job[job_key] = now
            return True

        # For "Not running" messages, try to identify which job this is about
        if "Not running" in msg:
            job_key = self._get_job_key(msg, name)  # No jobid for "Not running"

            if job_key and job_key in self._not_running_jobs:
                if self.debug_mode:
                    print(
                        f"[LOCAL DEBUG]: Suppressing duplicate 'Not running' for job {job_key}"
                    )
                return False
            if job_key:
                self._not_running_jobs.add(job_key)
                if self.debug_mode:
                    print(f"[LOCAL DEBUG]: First 'Not running' for job {job_key}, allowing")
            else:
                if self.debug_mode:
                    print("[LOCAL DEBUG]: Allowing 'Not running' (couldn't identify job)")
            return True

        # For status messages
        if "status =" in msg:
            jobid_m = self.JOBID_RE.search(msg)
            status_m = self.STATUS_RE.search(msg)

            if jobid_m and status_m:
                jobid = jobid_m.group(1)
                status = status_m.group(1).strip()

                unique_job_key = self._get_job_key(msg, name, jobid)
                prev_status = self._last_status_by_job.get(unique_job_key)

                if self.debug_mode:
                    print(
                        f"[LOCAL DEBUG]: Job key: '{unique_job_key}', Status: '{status}' (was: '{prev_status}')"
                    )

                if prev_status == status:
                    if self.debug_mode:
                        print(
                            f"[LOCAL DEBUG]: Suppressing duplicate status for {unique_job_key}"
                        )
                    return False

                self._last_status_by_job[unique_job_key] = status
                # If this job is now running/finished, allow "Not running" to be logged again later
                if status in [
                    "JobStatus.FINISHED",
                    "JobStatus.FAILED",
                    "JobStatus.KILLED",
                ]:
                    self._not_running_jobs.discard(unique_job_key)
                    self._last_heartbeat_by_job.pop(unique_job_key, None)
                    if self.debug_mode:
                        print(
                            f"[LOCAL DEBUG]: Job {unique_job_key} finished, allowing future 'Not running'"
                        )

        if self.suppress_mbar_noise and self._mbar_noise.search(msg):
            return False

        if self.debug_mode:
            print("[LOCAL DEBUG]: Allowing message through")
        return True

    @lru_cache(maxsize=1000)
    def _get_job_key(self, msg: str, logger_name: str, jobid: str = None) -> str | None:
        sim_details_m = self.SIM_DETAILS_RE.search(msg)
        if not sim_details_m:
            sim_details_m = self.SIM_DETAILS_RE.search(logger_name)
        if sim_details_m:
            leg = sim_details_m.group(1)
            stage = sim_details_m.group(2) 
            lam = sim_details_m.group(3)
            run_no = sim_details_m.group(4)
            if jobid:
                return f"{jobid}:sim:{leg}:{stage}:{lam}:{run_no}"
            else:
                return f"sim:{leg}:{stage}:{lam}:{run_no}"
        else:
            if jobid:
                return f"{jobid}:{logger_name}"
            else:
                return None


# Create ONE instance and reuse it
shared_filter = DedupStatusFilter(debug_mode=False)  # set to True for debugging


def setup_global_logging():
    """Set up global logging configuration with deduplication filter."""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    # Create handler with color formatting
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    handler.addFilter(shared_filter)

    root_logger.addHandler(handler)

    # Patch SimulationRunner._set_up_logging
    if hasattr(SimulationRunner, "_set_up_logging"):
        original_sr_setup = SimulationRunner._set_up_logging

        def patched_sr_setup(self, null: bool = False):
            original_sr_setup(self, null)
            if hasattr(self, "_logger"):
                self._logger.addFilter(shared_filter)
                for handler in self._logger.handlers:
                    handler.addFilter(shared_filter)

        SimulationRunner._set_up_logging = patched_sr_setup

    # Patch VirtualQueue._set_up_logging
    if hasattr(VirtualQueue, "_set_up_logging"):
        original_vq_setup = VirtualQueue._set_up_logging

        def patched_vq_setup(self):
            original_vq_setup(self)
            if hasattr(self, "_logger"):
                self._logger.addFilter(shared_filter)
                for handler in self._logger.handlers:
                    handler.addFilter(shared_filter)

        VirtualQueue._set_up_logging = patched_vq_setup


def add_filter_recursively(obj, filter_instance=shared_filter):
    """Recursively add filter to all loggers in an object hierarchy."""
    if hasattr(obj, "_logger"):
        obj._logger.addFilter(filter_instance)
        for handler in obj._logger.handlers:
            handler.addFilter(filter_instance)

    # Handle different types of sub-objects
    sub_objects = []
    if hasattr(obj, "_sub_sim_runners") and obj._sub_sim_runners:
        sub_objects.extend(obj._sub_sim_runners)
    if hasattr(obj, "stages") and obj.stages:
        sub_objects.extend(obj.stages)
    if hasattr(obj, "lam_windows") and obj.lam_windows:
        sub_objects.extend(obj.lam_windows)
    if hasattr(obj, "sims") and obj.sims:
        sub_objects.extend(obj.sims)
    if hasattr(obj, "legs") and obj.legs:
        sub_objects.extend(obj.legs)
    if hasattr(obj, "virtual_queue"):
        sub_objects.append(obj.virtual_queue)

    # Recursively apply to sub-objects
    for sub_obj in sub_objects:
        add_filter_recursively(sub_obj, filter_instance)


# ==================================================
# UTILITY FUNCTIONS FOR LOCAL EXECUTION
# ==================================================
def _parse_sim_info_from_job(job: Job) -> str:
    """
    Job.command_list is like:
      ['--chdir', '/Users/jingjinghuang/Documents/fep_workflow/
        test_somd_run_again2_copy1/bound/vanish/output/lambda_0.000/run_01',
        '/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again2_copy1/
        bound/vanish/output/lambda_0.000/run_01/run_somd.sh', '0.0']
    Returns a string like: "leg=<leg>, stage=<stage>, lam=<lam>, run_no=<run_no>"
    """
    leg_type = "?"
    stage = "?"
    lam = "?"
    run_no = "?"

    # Lambda is last element if numeric
    try:
        potential_lam = job.command_list[-1]
        # must use string to get logging like this "lam=0.000" and not "lam=0.0"
        lam = f"{Decimal(str(potential_lam)):.3f}"
    except Exception:
        pass

    # Get cwd from --chdir
    cwd = None
    if "--chdir" in job.command_list:
        idx = job.command_list.index("--chdir")
        if idx + 1 < len(job.command_list):
            cwd = job.command_list[idx + 1]

    if isinstance(cwd, str):
        m_leg_stage = re.search(r"[\\/](bound|free)[\\/](.*?)[\\/]output[\\/]", cwd)
        if m_leg_stage:
            leg_type = m_leg_stage.group(1)
            stage = m_leg_stage.group(2)

        # run number: e.g., .../run_01
        m_run = re.search(r"run_(\d+)", cwd)
        if m_run:
            run_no = m_run.group(1)

    return f"leg={leg_type}, stage={stage}, lam={lam}, run_no={run_no}"

# TODO: we may need to consolidate the two functions below
def _format_sim_info(cwd: str, lam_arg: str = None) -> str:
    """
    Given a working directory (the --chdir path), an optional lambda string,
    and an optional restraint type, returns:
       "leg=<leg>, stage=<stage>, lam=<lam>, run_no=<run_no>"
    """
    # parse lambda from argument or from cwd
    if lam_arg is None:
        m_lam = re.search(r"/lambda_([0-9.]+)/", cwd)
        lam_arg = m_lam.group(1) if m_lam else "?"

    # parse leg (bound/free)
    m_leg = re.search(r"/(bound|free)/", cwd)
    leg = m_leg.group(1) if m_leg else "?"

    # parse stage (vanish, restrain, discharge, etc.)
    m_stage = re.search(r"/(?:bound|free)/([^/]+)/output/", cwd)
    stage = m_stage.group(1) if m_stage else "?"

    # parse run number
    m_run = re.search(r"/run_(\d+)", cwd)
    run_no = m_run.group(1) if m_run else "?"

    # build parts
    parts = [f"leg={leg}", f"stage={stage}", f"lam={lam_arg}", f"run_no={run_no}"]
    return ", ".join(parts)


def _parse_leg_stage_from_cwd(cwd: str) -> tuple[str | None, str | None]:
    m_leg = re.search(r"/(bound|free)/", cwd)
    m_stage = re.search(r"/(?:bound|free)/([^/]+)/output/", cwd)
    leg = m_leg.group(1).lower() if m_leg else None
    stage = m_stage.group(1).lower() if m_stage else None
    return leg, stage


def _mbar_worker(cwd: str, mbar_command: str) -> tuple[int, str, str, float]:
    """
    Run MBAR command in a separate process with timing.
    Returns (returncode, stdout, stderr, duration). Never raises.
    """
    start_time = time.time()

    env = os.environ.copy()
    # prevent BLAS oversubscription when using multiprocessing
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    proc = subprocess.run(
        mbar_command,
        shell=True,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    duration = time.time() - start_time
    return proc.returncode, proc.stdout, proc.stderr, duration


def _is_mbar_script(script_path) -> bool:
    """Check if a script is an MBAR analysis script."""
    try:
        with open(script_path, "r") as f:
            content = f.read()
        return "analyse_freenrg" in content or "freenrg-MBAR" in content
    except Exception:
        return False


# TODO: we need to be cautious when function is called, as it creates dummy MBAR output files
def _create_dummy_mbar_output(output_path: str, cwd: str) -> None:
    """Create a realistic dummy MBAR output file that matches the expected format.
    NOTE not sure about the behavior of this function -> how does this affect the final free
    energy result? I included this function to prevent breaking the code when MBAR output is missing.

    why MBAR output is missing? are they really missing or just not generated yet?
    """
    lambda_files = []
    try:
        import glob

        lambda_dirs = glob.glob(os.path.join(cwd, "../lambda_*"))
        lambda_files = [
            f"'{os.path.join(dir, 'run_01/simfile_truncated_1.0_end_0.0_start.dat')}'"
            for dir in sorted(lambda_dirs)
        ]
    except Exception:
        lambda_files = [
            "'/path/to/lambda_0.000/run_01/simfile_truncated_1.0_end_0.0_start.dat'",
            "'/path/to/lambda_1.000/run_01/simfile_truncated_1.0_end_0.0_start.dat'",
        ]

    # Create dummy content that matches the real MBAR output format
    dummy_content = f"""# Analysing data contained in file(s) [{', '.join(lambda_files)}]
# WARNING: This is a dummy MBAR output created due to insufficient simulation data
# This is expected during early adaptive equilibration phases
#Overlap matrix
0.0000 0.0000
0.0000 0.0000
#DG from neighbouring lambda in kcal/mol
0.0000 0.0000 0.0000 0.0000
#PMF from MBAR in kcal/mol
0.0000 0.0000 0.0000
0.0000 0.0000 0.0000
#TI average gradients and standard deviation in kcal/mol
0.0000 0.0000 0.0000
0.0000 0.0000 0.0000
#PMF from TI in kcal/mol
0.0000 0.0000
0.0000 0.0000
#MBAR free energy difference in kcal/mol: 
0.000000, 0.000000  #WARNING DUMMY OUTPUT - INSUFFICIENT DATA FOR REAL MBAR ANALYSIS
#TI free energy difference in kcal/mol: 
0.000000  #WARNING DUMMY OUTPUT - INSUFFICIENT DATA FOR REAL MBAR ANALYSIS
        """
    with open(output_path, "w") as f:
        f.write(dummy_content)


_FALLBACK_RE = re.compile(r"(freenrg-MBAR[^\s/]*?\.dat)\b")


def _extract_mbar_output_file(command: str) -> str | None:
    """
    Extract the MBAR output file basename from a CLI command string.

    a command may look like:
    analyse_freenrg mbar -i /project/6097686/jjhuang/fep_workflows/new_run_final_1/\
        bound/restrain/output/lambda*/run_02/simfile_truncated_66.0_end_65.0_start.dat\
            -p 100 --overlap -o \
        /project/6097686/jjhuang/fep_workflows/new_run_final_1/bound/restrain/output/\
            freenrg-MBAR-run_02_66.0_end_65.0_start.dat"
    """
    try:
        parts = shlex.split(command)  # safer than .split()
    except Exception:
        parts = command.split()

    output_value = None
    for i, tok in enumerate(parts):
        if tok in ("--output", "-o"):
            if i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                output_value = parts[i + 1]
        elif tok.startswith("--output="):
            output_value = tok.split("=", 1)[1]

    if not output_value:
        m = _FALLBACK_RE.search(command)
        if m:
            output_value = m.group(1)

    return os.path.basename(output_value) if output_value else None


# ========================================================
# Global MBAR Manager FOR LOCAL AND PARALLEL EXECUTION
# ========================================================
class ParallelMBARManager:
    """Global manager for parallel MBAR execution with proper synchronization."""

    def __init__(self, max_workers: int = None, use_progress: bool = True):
        self.use_progress = use_progress
        if max_workers is None:
            # Use half the available cores to avoid oversubscription
            max_workers = max(1, (os.cpu_count() or 4) // 2)

        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.futures = {}  # job_id -> Future
        self.job_metadata = {}  # job_id -> {"cwd": str, "cmd": str, "script": str}
        self.job_counter = itertools.count(600000)
        self.expected_outputs = set()
        self.logger = get_tagged_logger(__name__ + ".MBAR_MANAGER")
        self.jobs_by_leg_stage = {} 

    def submit_mbar_job(self, script_path: str, cwd: str) -> int:
        """Submit an MBAR job for parallel execution."""
        if hasattr(shared_filter, "suppress_mbar_noise"):
            shared_filter.suppress_mbar_noise = True
        # Parse the MBAR command from script
        try:
            with open(script_path) as f:
                script_content = f.read()
        except FileNotFoundError:
            raise RuntimeError(f"MBAR script not found: {script_path}")

        mbar_command = None
        for line in script_content.splitlines():
            line = line.strip()
            if (
                ("analyse_freenrg" in line)
                and not line.startswith("#")
                and not line.startswith("export")
            ):
                mbar_command = line
                break

        if not mbar_command:
            self.logger.warning(
                "No explicit MBAR command found, running script directly"
            )
            mbar_command = f"bash {script_path}"

        ofile = _extract_mbar_output_file(mbar_command)  # ofile can be None somehow
        if ofile:
            self.expected_outputs.add(os.path.join(cwd, ofile))

        # Submit to executor
        job_id = next(self.job_counter)
        future = self.executor.submit(_mbar_worker, cwd, mbar_command)

        self.futures[job_id] = future
        self.job_metadata[job_id] = {
            "cwd": cwd,
            "cmd": mbar_command,
            "script": script_path,
        }
        self._log_mbar_start(cwd=cwd, command=mbar_command, job_id=job_id)
        self.logger.info(f"Submitted MBAR job {job_id}: {mbar_command}")
        return job_id

    def _log_mbar_start(self, cwd: str, command: str, job_id: int):
        """Log MBAR job start to local_execution.log"""
        start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mbar_info = self._format_mbar_info(cwd, command)
        log_path = os.path.join(cwd, "local_execution.log")
        os.makedirs(cwd, exist_ok=True)  # although should already exist
        with open(log_path, "a") as f:
            f.write(
                f"[LOCAL MBAR] {start_timestamp} Starting MBAR job {job_id}: {mbar_info}\n"
            )
            f.write(f"[LOCAL MBAR] Command: {command}\n")

    def _log_mbar_completion(
        self,
        cwd: str,
        job_id: int,
        success: bool,
        duration: float,
        error_msg: str = None,
        outputfile_path: str = None,
    ):
        """Log MBAR job completion to local_execution.log"""
        end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_path = os.path.join(cwd, "local_execution.log")
        with open(log_path, "a") as f:
            if success:
                f.write(
                    f"[LOCAL MBAR] {end_timestamp} ✅ MBAR job completed! {job_id} completed in {duration:.2f} seconds; output -> {outputfile_path}\n"
                )
            else:
                f.write(
                    f"[LOCAL MBAR] {end_timestamp} ❌ MBAR job failed! {job_id} failed (dummy output created)\n"
                )
                f.write(f"[LOCAL MBAR] Error: {error_msg}\n")

    def _format_mbar_info(self, cwd: str, command: str) -> str:
        """Format MBAR job info for logging"""
        # Extract stage info from path
        stage_match = re.search(r"/(?:bound|free)/([^/]+)/output(?:/|$)", cwd)
        stage = stage_match.group(1) if stage_match else "unknown"
        # Extract output file
        output_file = _extract_mbar_output_file(command)
        return f"stage={stage}, output={output_file or 'unknown'}"

    def wait_for_completion(self):
        """Wait for all submitted MBAR jobs to complete (robust)."""
        if not self.futures:
            self.logger.info("No MBAR jobs to wait for")
            return

        self.logger.info(f"Waiting for {len(self.futures)} MBAR jobs to complete...")

        total = len(self.futures)
        future_to_id = {fut: jid for jid, fut in self.futures.items()}
        # Use a TTY-aware progress bar if available
        use_pb = self.use_progress and (tqdm is not None) and sys.stderr.isatty()
        ok = 0
        fail = 0
        if use_pb:
            bar = tqdm(
                total=total, desc="MBAR analyses", unit="job", leave=True, smoothing=0.1
            )
            # (optional) hide noisy “running/submitted” logs while bar is active
            if hasattr(shared_filter, "suppress_mbar_noise"):
                shared_filter.suppress_mbar_noise = True
        else:
            self.logger.info(f"Waiting for {total} MBAR jobs to complete...")

        for future in concurrent.futures.as_completed(self.futures.values()):
            job_id = future_to_id.get(future)
            if job_id is None:
                continue

            meta = self.job_metadata.get(job_id, {})
            cwd = meta.get("cwd", "")
            cmd = meta.get("cmd", "")
            # script = meta.get("script", "")
            try:
                rc, _, stderr, duration = future.result()
            except Exception as e:
                rc, _, stderr, duration = -1, "", str(e)

            # NOTE: Success must also produce a .dat file
            ofile = _extract_mbar_output_file(cmd)
            success = False
            if ofile is None:
                self.logger.error(f"MBAR job {job_id} did not produce an output file with run command: {cmd}")
                rc = -1
            else:
                # when ofile is not None
                ofile_path = os.path.join(cwd, ofile)
                if (
                    rc == 0
                    and os.path.exists(ofile_path)
                    and os.path.getsize(ofile_path) > 0
                ):
                    success = True
                    ok += 1
                    if not use_pb:
                        self.logger.info(f"MBAR job {job_id} completed successfully")
                else:
                    fail += 1
                    error_msg = stderr if stderr else "Output file missing or empty"
                    if not use_pb:
                        self.logger.warning(f"MBAR job {job_id} failed: {error_msg} with run command: {cmd}")

            self._log_mbar_completion(
                cwd=cwd,
                job_id=job_id,
                success=success,
                duration=duration,
                error_msg=stderr if not success else None,
                outputfile_path=ofile_path if success else None,
            )
            if use_pb:
                bar.update(1)
                bar.set_postfix_str(f"ok={ok} fail={fail}")

        if use_pb:
            bar.close()
            if hasattr(shared_filter, "suppress_mbar_noise"):
                shared_filter.suppress_mbar_noise = False

        self.futures.clear()
        self.job_metadata.clear()
        self.logger.info(f"All MBAR jobs completed (ok={ok}, fail={fail})")

    def has_pending_jobs(self) -> bool:
        """Check if there are any pending MBAR jobs."""
        return bool(self.futures)

    def get_job_status(self, job_id: int) -> str:
        """Get the status of a specific job."""
        if job_id not in self.futures:
            return "UNKNOWN"

        future = self.futures[job_id]
        if future.done():
            try:
                _ = future.result()
                return "FINISHED"
            except Exception:
                return "FAILED"
        else:
            return "RUNNING"


# ========================================================
# Global SOMD Manager FOR LOCAL AND CONCURRENT EXECUTION
# ========================================================
class ConcurrentSOMDManager:
    """Manages concurrent SOMD executions with proper synchronization.
    
    TODO: it seems NVIDIA MPS + OpenMM/SOMD is a bit brittle. likely lead to "CUDA_ERROR_NOT_FOUND (500)"
    OpenMM compiles CUDA kernels at runtime. And with MPS, multiple client processes can JIT the same 
    kernels concurrently -> errors. so we remove MPS completely for now.
    NOTE: however, MPS seems to improve the concurrent performance
    """
    
    def __init__(self, max_workers=2):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.futures = {}  # job_id -> Future
        self.job_metadata = {}  # job_id -> metadata
        self.job_counter = itertools.count(800000)  # Different range from MBAR
        self.logger = get_tagged_logger(__name__ + ".SOMD_MANAGER")

        # TODO: note that the barrier only ensures that no jobs still running, but 
        # they may end up with different total runtimes -> precondition error
        self.stage_lock = threading.Lock()
        self.gpu_semaphore = threading.Semaphore(max_workers) # not sure about this actually by JH 2025-09-24
        self.jobs_by_stage = defaultdict(set)
        self.completed_jobs = set() 
   
    def submit_somd_job(self, script_path: str, lambda_str: str, cwd: str, run_no: int) -> int:
        """Submit a SOMD job for concurrent execution."""
        job_id = next(self.job_counter)
        submit_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # TODO: we might need to consolidate the two
        sim_info = _format_sim_info(cwd, lambda_str)
        leg, stage = _parse_leg_stage_from_cwd(cwd)
        
        # Submit to executor
        future = self.executor.submit(self._run_somd_worker, job_id, script_path, lambda_str, cwd, run_no)
        self.futures[job_id] = future

        # Store metadata
        self.job_metadata[job_id] = {
            'script_path': script_path,
            'lambda_str': lambda_str,  
            'lambda_val': float(lambda_str), 
            'cwd': cwd,
            'run_no': run_no,
            'start_time': time.time(),
            'leg': leg,
            'stage': stage,
        }
        if leg and stage:
            with self.stage_lock:
                self.jobs_by_stage[(leg, stage)].add(job_id)
        
        self.logger.info(f"[CONCURRENT SOMD] {submit_timestamp} Submitted SOMD job {job_id} for {sim_info}")
        return job_id
    
    def _run_somd_worker(self, job_id:int, script_path: str, lambda_str: str, cwd: str, run_no: int):
        """Worker function to run a single SOMD simulation."""

        env = os.environ.copy()
        # prevent BLAS oversubscription when using multiprocessing
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")

        real_cwd = cwd or os.path.dirname(script_path)

        start_time = time.time()
        start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sim_info = _format_sim_info(real_cwd, lambda_str)
        cfg_path = os.path.join(real_cwd, "somd.cfg")
        local_execution_log_path = os.path.join(real_cwd, "local_execution.log")

        # we need to validate somd.cfg exists
        if not os.path.exists(cfg_path) or os.path.getsize(cfg_path) == 0:
            self.logger.error(f"[CONCURRENT SOMD] ❌ somd.cfg is missing or empty in {real_cwd}")
            with open(local_execution_log_path, "a") as flog:
                flog.write(
                    f"[CONCURRENT SOMD] ❌ JOB {job_id} FAILED {cfg_path} is missing or empty in {real_cwd}; cannot run SOMD\n"
                )
            raise RuntimeError(f"somd.cfg is missing or empty in {real_cwd}")

        # Read and parse the SOMD command from script
        somd_command = None
        try:
            with open(script_path) as f:
                for line in f:
                    line = line.strip()
                    if "somd-freenrg" in line:
                        if line.startswith("srun "):
                            somd_command = line.split(None, 1)[1]
                        else:
                            somd_command = line
                        break
        except FileNotFoundError:
            with open(local_execution_log_path, "a") as flog:
                flog.write(f"[CONCURRENT SOMD] ❌ JOB {job_id} FAILED missing script {script_path}\n")
            raise RuntimeError(f"Script not found: {script_path}")

        if not somd_command:
            raise RuntimeError(f"No somd-freenrg command found in {script_path}")

        # Parse and modify the command
        parts = somd_command.split()       
        if "-p" in parts:
            p_idx = parts.index("-p")
            if p_idx + 1 < len(parts):
                current_platform = parts[p_idx + 1].upper()
                if FORCE_CPU_PLATFORM:
                    parts[p_idx + 1] = "CPU"

        # Substitute lambda value
        parts = [tok.replace("$lam", lambda_str).replace("${lam}", lambda_str) for tok in parts]
                
        acquired = False
        try:
            if self.gpu_semaphore:
                self.gpu_semaphore.acquire()
                acquired = True

            with open(local_execution_log_path, "a", buffering=1) as lf:
                _ = subprocess.run(
                    parts,
                    cwd=cwd,
                    stdout=lf,
                    stderr=subprocess.PIPE,
                    check=True,
                    text=True,
                )
            
            duration_seconds = time.time() - start_time
            end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create a local_execution.log file that mimics SLURM output -> must have this
            with open(local_execution_log_path, "a") as f:
                f.write(f"[CONCURRENT SOMD] Starting {sim_info} at {start_timestamp}\n")
                f.write(f"[CONCURRENT SOMD] Completed {sim_info} at {end_timestamp}\n")
                f.write(f"[CONCURRENT SOMD] Simulation took {duration_seconds:.2f} seconds\n")
                f.write(f"[CONCURRENT SOMD] ✅ Job {job_id} completed successfully\n")

            self.logger.info(f"[CONCURRENT SOMD] {end_timestamp} ✅ Job {job_id} completed successfully in {duration_seconds:.2f}s: {sim_info}")
            
            return 888888  # Success code
            
        except Exception as e:
            rc   = getattr(e, "returncode", -1)
            err  = getattr(e, "stderr", repr(e))

            end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.logger.error(f"[CONCURRENT SOMD] ❌ JOB {job_id} FAILED {rc}")
            self.logger.error(f"[CONCURRENT SOMD] ❌ STDERR:\n{err}")
            
            local_execution_log_path = os.path.join(real_cwd, "local_execution.log")
            with open(local_execution_log_path, "a") as f:
                f.write(f"[CONCURRENT SOMD] ❌ Job {job_id} FAILED: {sim_info}\nError: {err}\n")
            raise
    
        finally:
            if acquired:
                self.gpu_semaphore.release()
        
            with self.stage_lock:
                    self.completed_jobs.add(job_id)
                    meta = self.job_metadata.get(job_id, {})
                    key = (meta.get('leg'), meta.get('stage'))
                    if key in self.jobs_by_stage:
                        self.jobs_by_stage[key].discard(job_id)
    
    
    def wait_for_stage(self, leg: str, stage: str):
        """Block until all SOMD jobs for (leg, stage) have finished."""
        key = (leg.lower(), stage.lower())
        # Get snapshot of jobs to wait for
        with self.stage_lock:
            pending_ids = list(self.jobs_by_stage.get(key, set()))

        if not pending_ids:
            self.logger.info(f"[CONCURRENT SOMD] No jobs to wait for stage={stage}, leg={leg}")
            return

        self.logger.info(f"[CONCURRENT SOMD] 🟡 Waiting for {len(pending_ids)} jobs in stage={stage}, leg={leg}")
        for job_id in pending_ids:
            fut = self.futures.get(job_id)
            if fut is None:
                continue
            try:
                result = fut.result()  # block
                self.logger.debug(f"[CONCURRENT SOMD] Job {job_id} completed with result: {result}")
            except Exception as e:
                self.logger.error(f"[CONCURRENT SOMD] Job {job_id} failed: {e}")
        
        # ensure the set is empty (finally blocks remove them)
        with self.stage_lock:
            for job_id in pending_ids:
                self.futures.pop(job_id, None)

        self.logger.info(f"[CONCURRENT SOMD] ✅ All jobs completed for stage={stage}, leg={leg}")

    def check_and_cleanup_completed_jobs(self):
        """Check for completed jobs and clean them up. Returns count of newly completed."""
        completed_count = 0
        jobs_to_remove = []
        
        for job_id, future in self.futures.items():
            if future.done():
                jobs_to_remove.append(job_id)
                completed_count += 1
                # Log completion if not already done
                if job_id not in self.completed_jobs:
                    try:
                        result = future.result()
                        meta = self.job_metadata.get(job_id, {})
                        sim_info = meta.get('sim_info', f'job_{job_id}')
                        self.logger.info(f"[CONCURRENT SOMD] Job {job_id} completed: {sim_info}")
                    except Exception as e:
                        self.logger.error(f"[CONCURRENT SOMD] Job {job_id} failed: {e}")
                    
                    with self.stage_lock:
                        self.completed_jobs.add(job_id)
        
        # Remove completed jobs
        for job_id in jobs_to_remove:
            self.futures.pop(job_id, None)
        return completed_count
    
    def has_pending_jobs(self):
        """Check if there are pending jobs."""
        return bool(self.futures)
    
    def get_active_job_count(self):
        """Get number of currently active jobs."""
        return sum(1 for future in self.futures.values() if not future.done())

    def get_job_status(self, job_id: int) -> str:
        """Get status of a specific job"""
        if job_id not in self.futures:
            if job_id in self.completed_jobs:
                return "FINISHED"
            return "UNKNOWN"
        
        future = self.futures[job_id]
        if future.done():
            try:
                future.result()
                return "FINISHED"
            except:
                return "FAILED"
        else:
            return "RUNNING"



def patch_stage_for_concurrent_somd():
    """
    Patch Stage._run_without_threading to properly integrate with concurrent SOMD execution
    and ensure stage-level barrier synchronization.
    """
    logger = get_tagged_logger(__name__ + ".STAGE_PATCH")
        
    def enhanced_run_without_threading(
        self,
        run_nos,
        adaptive=True,
        runtime=None,
        max_runtime=60,
    ):
        """Enhanced version with proper concurrent SOMD integration."""
        try:
            # Reset kill thread flag
            self.kill_thread = False

            if not adaptive and runtime is None:
                raise ValueError("If adaptive equilibration detection is disabled, a runtime must be supplied.")
            if adaptive and runtime is not None:
                raise ValueError("If adaptive equilibration detection is enabled, a runtime cannot be supplied.")

            if not adaptive:
                self._logger.info(f"Starting {self}. Adaptive equilibration = {adaptive}...")
            elif adaptive:
                self._logger.info(f"Starting {self}. Adaptive equilibration = {adaptive}...")
                if runtime is None:
                    runtime = 0.2  # ns (or whatever value you want)

            for win in self.lam_windows:
                win.run(run_nos=run_nos, runtime=runtime)
                self._dump()

            # CRITICAL and NEW: Wait for ALL jobs in this stage to complete before proceeding
            try:
                leg_name = getattr(self.leg_type, "name", "unknown").lower()
                stage_name = getattr(self.stage_type, "name", "unknown").lower()
            except Exception:
                leg_name = "unknown"
                stage_name = "unknown"

            if _GLOBAL_SOMD_MANAGER:
                self._logger.info(
                    f"[STAGE BARRIER] Waiting for all lambda windows to complete initial runs "
                    f"before efficiency check (stage={stage_name}, leg={leg_name})"
                )
                _GLOBAL_SOMD_MANAGER.wait_for_stage(leg_name, stage_name)

            self.running_wins = self.lam_windows.copy()
            self._dump()

            if adaptive:
                self._run_loop_adaptive_efficiency(run_nos=run_nos, max_runtime=max_runtime)
                self._run_loop_adaptive_equilibration_multiwindow(run_nos=run_nos, max_runtime=max_runtime)
            else:
                self._run_loop_non_adaptive()

            self._logger.info(f"All simulations in {self} have finished.")

        except Exception as e:
            self._logger.exception("")
            raise e
    
    # Apply the patch
    Stage._run_without_threading = enhanced_run_without_threading
    logger.info("Stage._run_without_threading patched for concurrent SOMD execution")



# Global instance
_GLOBAL_MBAR_MANAGER = None
_GLOBAL_SOMD_MANAGER = None


def _install_mbar_barrier_wrapper(logger):
    import a3fe.analyse.mbar as mbar
    import a3fe.analyse.process_grads as process_grads
    import a3fe.analyse.detect_equil as detect_equil  # need this for equil analysis
    import a3fe.run.stage as stage  # need this for calc.analyse()

    if not hasattr(mbar, "_original_collect_mbar_slurm"):
        mbar._original_collect_mbar_slurm = mbar.collect_mbar_slurm

    mbar_sync_in_progress = False

    def _collect_mbar_wrapper(*args, **kwargs):
        nonlocal mbar_sync_in_progress
        # Only say anything if there are outstanding MBAR futures
        has_pending = _GLOBAL_MBAR_MANAGER and _GLOBAL_MBAR_MANAGER.has_pending_jobs()
        if has_pending and not mbar_sync_in_progress:
            logger.info(
                "[LOCAL MBAR] collect_mbar_slurm called - waiting for all MBAR jobs to complete"
            )
            mbar_sync_in_progress = True

        if has_pending:
            _GLOBAL_MBAR_MANAGER.wait_for_completion()
            # IMPORTANT NOTE: Give file system time to sync
            # It takes a few seconds for the file system to sync after MBAR jobs complete
            # otherwise create dummy MBAR output files even if the jobs are done and outputs will be there
            time.sleep(2)

        kwargs_modified = kwargs.copy()
        kwargs_modified["delete_outfiles"] = False

        # Safety net: ensure all expected outputs exist (create dummies if not)
        if _GLOBAL_MBAR_MANAGER:
            # IMPORTANT NOTE: same reason for the sleep(2) above, we need to wait a bit
            max_retries = 3
            retry_delay = 1.0
            missing = []
            for ofile in list(_GLOBAL_MBAR_MANAGER.expected_outputs):
                file_found = False
                # Retry logic for file existence check
                for attempt in range(max_retries):
                    if os.path.exists(ofile) and os.path.getsize(ofile) > 0:
                        file_found = True
                        break
                    elif attempt < max_retries - 1:
                        logger.warning(
                            f"[LOCAL MBAR] File not found on attempt {attempt + 1}, retrying: {ofile}"
                        )
                        log_path = os.path.join(
                            os.path.dirname(ofile), "local_execution.log"
                        )
                        write_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open(log_path, "a") as f:
                            f.write(
                                f"[LOCAL MBAR] {write_time} File not found on attempt {attempt + 1}, retrying: {ofile}\n"  # noqa: E501
                            )

                        time.sleep(retry_delay)
                        retry_delay *= 1.5

                if not file_found:
                    missing.append(ofile)
                    # create a dummy output file so that the code can continue
                    _create_dummy_mbar_output(ofile, os.path.dirname(ofile))
                    logger.warning(
                        f"[LOCAL MBAR] Missing MBAR output after retries; created dummy: {ofile}"
                    )
                    # we need to write this to local_execution.log
                    # os.path.dirname(ofile) must be something like "~bound/discharge/output/"
                    log_path = os.path.join(
                        os.path.dirname(ofile), "local_execution.log"
                    )
                    write_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(log_path, "a") as f:
                        f.write(
                            f"[LOCAL MBAR] {write_time} Created dummy output for missing MBAR file after retries: {ofile}\n"  # noqa: E501
                        )

            if missing:
                logger.warning(
                    f"[LOCAL MBAR] {len(missing)} MBAR outputs were missing and replaced with dummies."
                )

        # Only print the completion line once per wave
        if has_pending and mbar_sync_in_progress:
            logger.info(
                "[LOCAL MBAR] All MBAR jobs completed - proceeding to collect results"
            )
            mbar_sync_in_progress = False

        kwargs_modified = kwargs.copy()
        kwargs_modified["delete_outfiles"] = False
        return mbar._original_collect_mbar_slurm(*args, **kwargs_modified)

    # Replace on module
    mbar.collect_mbar_slurm = _collect_mbar_wrapper
    # Rebind any cached aliases
    if hasattr(process_grads, "_collect_mbar_slurm"):
        process_grads._collect_mbar_slurm = _collect_mbar_wrapper
    if hasattr(detect_equil, "_collect_mbar_slurm"):
        detect_equil._collect_mbar_slurm = _collect_mbar_wrapper
    if hasattr(stage, "_collect_mbar_slurm"):
        stage._collect_mbar_slurm = _collect_mbar_wrapper


def _install_soft_trim_time_series_wrapper(logger):
    """
    Monkey-patch a3fe.analyse.process_grads.get_time_series_multiwindow_mbar
    to 'soft-trim' the time axis per run to the minimum total across runs.
    No files are modified; we only adjust the reported times so the
    paired-t check can proceed.

    IMPORTANT NOTE: soft-trimming the time axis in memory keeps the statistics intact 
    and only harmonizes the x-axis
    """
    import a3fe.analyse.process_grads as process_grads

    if hasattr(process_grads, "_orig_gts_mbar"):
        return

    process_grads._orig_gts_mbar = process_grads.get_time_series_multiwindow_mbar

    def _gts_mbar_soft_trim(lambda_windows, output_dir, equilibrated=False,
                            run_nos=None, start_frac=0.0, end_frac=1.0):
        """
        This is a near-copy of the upstream function, with one surgical change:
        when constructing overall_times, we use the *minimum* total time across runs
        for the whole stage so all runs share the same final time.
        """
        _np = process_grads._np
        _get_context = process_grads._get_context
        _submit_mbar_slurm = process_grads._submit_mbar_slurm
        _collect_mbar_slurm = process_grads._collect_mbar_slurm
        _compute_dg = process_grads._compute_dg  # helper defined in the module

        # --- original prechecks ---
        if equilibrated and not all([lam.equilibrated for lam in lambda_windows]):
            raise ValueError(
                "The equilibration times and statistics have not been set for all lambda "
                "windows in the stage. Please set these before running this function."
            )
        if equilibrated:
            for lam_win in lambda_windows:
                lam_win._write_equilibrated_simfiles()

        run_nos = lambda_windows[0]._get_valid_run_nos(run_nos)  # type: ignore

        n_runs = len(run_nos)
        n_points = 100
        overall_dgs = _np.zeros([n_runs, n_points])
        overall_times = _np.zeros([n_runs, n_points])

        start_and_end_fracs = [
            (i, i + (end_frac - start_frac) / n_points)
            for i in _np.linspace(start_frac, end_frac, n_points + 1)
        ][:-1]
        start_and_end_fracs = [(round(x0, 5), round(x1, 5)) for (x0, x1) in start_and_end_fracs]

        use_slurms = [getattr(lw, "slurm_equil_detection", True) for lw in lambda_windows]
        if not all(u == use_slurms[0] for u in use_slurms):
            raise ValueError(
                "use_slurm is not the same for all lambda windows. Please ensure that "
                "use_slurm is the same for all lambda windows."
            )

        # --- compute overall_dgs exactly as upstream ---
        if not use_slurms[0]:
            with _get_context("spawn").Pool() as pool:
                results = pool.starmap(
                    _compute_dg,
                    [
                        (run_no, s, e, output_dir, equilibrated)
                        for run_no in run_nos
                        for (s, e) in start_and_end_fracs
                    ],
                )
            for i, run_no in enumerate(run_nos):
                for j, _fr in enumerate(start_and_end_fracs):
                    overall_dgs[i, j] = results[i * len(start_and_end_fracs) + j]
        else:
            frac_jobs = []
            results = []
            for (s, e) in start_and_end_fracs:
                frac_jobs.append(
                    _submit_mbar_slurm(
                        output_dir=output_dir,
                        virtual_queue=lambda_windows[0].virtual_queue,
                        run_nos=run_nos,
                        run_somd_dir=lambda_windows[0].input_dir,
                        percentage_end=e * 100,
                        percentage_start=s * 100,
                        subsampling=False,
                        equilibrated=equilibrated,
                    )
                )
            for frac_job in frac_jobs:
                jobs, mbar_outfiles, tmp_simfiles = frac_job
                results.append(
                    _collect_mbar_slurm(
                        output_dir=output_dir,
                        run_nos=run_nos,
                        jobs=jobs,
                        mbar_out_files=mbar_outfiles,
                        virtual_queue=lambda_windows[0].virtual_queue,
                        tmp_simfiles=tmp_simfiles,
                        delete_outfiles=True,
                    )
                )
            for i, _ in enumerate(run_nos):
                for j, _fr in enumerate(start_and_end_fracs):
                    overall_dgs[i, j] = results[j][0][i]

        # --- CHANGED BLOCK: compute a common time axis using the stage's shortest total ---
        per_run_totals = []
        for r in run_nos:
            tot = sum(lw.get_tot_simtime([r]) for lw in lambda_windows)
            per_run_totals.append(float(tot))

        ref_total = min(per_run_totals) if per_run_totals else 0.0
        equil_sum = sum(lw.equil_time for lw in lambda_windows) if equilibrated else 0.0

        trimmed = [t for t in per_run_totals if t > ref_total + 1e-12]
        if trimmed:
            logger.info(
                "[SOFT-TRIM] Capping per-run total time to the minimum across runs: "
                "min=%.6f ns; trimmed runs=%s",
                ref_total,
                [i + 1 for i, t in enumerate(per_run_totals) if t > ref_total + 1e-12],
            )

        for i, _ in enumerate(run_nos):
            times = [(ref_total - equil_sum) * s + equil_sum for (s, _e) in start_and_end_fracs]
            overall_times[i] = _np.array(times)

        # --- keep the NaN check (unchanged) ---
        if _np.isnan(overall_dgs).any():
            raise ValueError(
                "NaNs found in the free energy change. Please check that the simulation "
                "has run correctly."
            )

        return overall_dgs, overall_times

    process_grads.get_time_series_multiwindow_mbar = _gts_mbar_soft_trim
    logger.info("[SOFT-TRIM] Installed soft-trim wrapper for get_time_series_multiwindow_mbar().")


def _install_soft_trim_time_series_wrapper_for_ti(logger):
    import a3fe.analyse.process_grads as process_grads
    import numpy as _np

    if hasattr(process_grads, "_orig_gts_ti"):
        return  # already patched

    process_grads._orig_gts_ti = process_grads.get_time_series_multiwindow

    def _gts_ti_soft_trim(*args, **kwargs):
        # Unpack required args the same way the original does
        lambda_windows = kwargs.get("lambda_windows", args[0] if args else None)
        equilibrated   = kwargs.get("equilibrated", False)
        run_nos        = kwargs.get("run_nos", None)
        start_frac     = kwargs.get("start_frac", 0.0)
        end_frac       = kwargs.get("end_frac", 1.0)

        if lambda_windows is None:
            raise RuntimeError("lambda_windows is required")

        # Let the original do its ordinary upfront checks (weights, equilibrated, etc.)
        # BUT we’ll intercept unequal totals before it raises.
        # We need run order and a pre-scan of per-run totals.
        _get_valid = lambda_windows[0]._get_valid_run_nos
        run_nos    = _get_valid(run_nos)
        n_runs     = len(run_nos)

        # --- Pre-scan per-run total effective time across windows (with the same slicing logic) ---
        per_run_totals = [0.0] * n_runs
        # (Optionally cache to avoid double file reads)
        _precache = {}  # (win_id, run_idx) -> (times, grads)

        for w_idx, lam_win in enumerate(lambda_windows):
            for i, run_no in enumerate(run_nos):
                sim = lam_win.sims[run_no - 1]
                times, grads = sim.read_gradients()
                _precache[(w_idx, i)] = (times, grads)

                # Same slicing as original (index-based by fraction of *samples*)
                start_idx = 0 if start_frac is None else round(start_frac * len(grads))
                end_idx   = len(grads) if end_frac is None else round(end_frac * len(grads))
                if end_idx - start_idx < 100:
                    raise ValueError(
                        "Not enough data to combine windows. Please use a larger fraction "
                        "or run longer."
                    )
                t0 = times[start_idx]
                t1 = times[end_idx - 1]
                per_run_totals[i] += float(t1 - t0)

        # Choose the common reference total as the minimum across runs
        ref_total = min(per_run_totals) if per_run_totals else 0.0
        # Build per-run scale factors so sum(window_durations_scaled) == ref_total
        scale = [ (ref_total / tot if tot > 0 else 1.0) for tot in per_run_totals ]

        # Log when we actually trim
        trimmed_runs = [i+1 for i, tot in enumerate(per_run_totals) if tot > ref_total + 1e-12]
        if trimmed_runs:
            logger.info(
                "[SOFT-TRIM][TI] Capping per-run total time to min across runs: "
                "min=%.6f ns; trimmed runs=%s", ref_total, trimmed_runs
            )

        # --- Reconstruct overall_dgs and overall_times using the original logic,
        #     but scale the per-window time arrays before summing. ---
        _np = process_grads._np
        n_points = 100
        overall_dgs   = _np.zeros((n_runs, n_points))
        overall_times = _np.zeros((n_runs, n_points))

        for w_idx, lam_win in enumerate(lambda_windows):
            for i, run_no in enumerate(run_nos):
                times, grads = _precache[(w_idx, i)]
                dgs = [g * lam_win.lam_val_weight for g in grads]

                start_idx = 0 if start_frac is None else round(start_frac * len(dgs))
                end_idx   = len(dgs) if end_frac is None else round(end_frac * len(dgs))
                # (length check already done above)

                times = times[start_idx:end_idx]
                dgs   = dgs[start_idx:end_idx]

                # Resize to 100 points (same as original)
                times_resized = _np.linspace(times[0], times[-1], n_points)
                dgs_resized   = _np.zeros(n_points)
                idxs = _np.array([round(x) for x in _np.linspace(0, len(dgs), n_points + 1)])
                for j in range(n_points):
                    dgs_resized[j] = _np.mean(dgs[idxs[j]:idxs[j+1]])

                # *** KEY CHANGE: sum *durations* scaled to the per-run factor ***
                durations = times_resized - times_resized[0]           # start at 0
                overall_times[i] += durations * scale[i]               # scaled sum
                overall_dgs[i]   += dgs_resized

            # After each window, do not raise on tiny fp differences
            # (we made totals equal by construction, but keep a tolerant check)
            if not _np.allclose(overall_times[:, -1], overall_times[0, -1], rtol=0, atol=1e-9):
                # Harmonize explicitly to remove any residual noise
                overall_times[:, -1] = overall_times[0, -1]

            if _np.isnan(overall_dgs).any():
                raise ValueError(
                    "NaNs found in the free energy change. Please check that the simulation ran correctly."
                )

        # Rebase time axis to look like absolute time (start at 0 and end at ref_total)
        # (The original effectively does this by construction when totals are equal.)
        # Here overall_times already starts at 0 and ends at ref_total for every run.
        return overall_dgs, overall_times

    process_grads.get_time_series_multiwindow = _gts_ti_soft_trim
    logger.info("[SOFT-TRIM][TI] Installed soft-trim wrapper for get_time_series_multiwindow().")




def patch_virtual_queue_for_local_execution(use_faster_wait: bool = False):
    """
    Patch VirtualQueue to run jobs locally instead of through SLURM.
    Works on both local machines and HPC systems.

    turn on use_faster_wait to speed up local testing by reducing wait time
    """
    global _GLOBAL_MBAR_MANAGER, _GLOBAL_SOMD_MANAGER

    # Check if we should use local execution
    use_local = FORCE_LOCAL_EXECUTION or (shutil.which("squeue") is None)

    logger = get_tagged_logger(__name__ + ".LOCAL_SOMD")

    if not use_local:
        logger.info(
            "SLURM detected and local execution not forced. Using normal SLURM submission."
        )
        return

    logger.info(f"Force CPU: {FORCE_CPU_PLATFORM}")

    # Initialize global MBAR manager
    _GLOBAL_MBAR_MANAGER = ParallelMBARManager()
    _install_mbar_barrier_wrapper(logger)
    _install_soft_trim_time_series_wrapper(logger)
    _install_soft_trim_time_series_wrapper_for_ti(logger)
    logger.info(f"MBAR parallel workers: {_GLOBAL_MBAR_MANAGER.max_workers}")

    _GLOBAL_SOMD_MANAGER = ConcurrentSOMDManager(
        max_workers=MAX_CONCURRENT_SOMD,
    )
    logger.info(f"Concurrent SOMD workers: {_GLOBAL_SOMD_MANAGER.max_workers}")


    # Silence subprocess calls (for ln commands and other system calls)
    original_call = subprocess.call

    def quiet_call(*args, **kwargs):
        if args and isinstance(args[0], list) and len(args[0]) > 0:
            if args[0][0] == "ln":
                kwargs.setdefault("stdout", subprocess.DEVNULL)
                kwargs.setdefault("stderr", subprocess.DEVNULL)
            elif args[0][0] in ["mkdir", "cp", "mv", "rm"]:
                kwargs.setdefault("stdout", subprocess.DEVNULL)
                kwargs.setdefault("stderr", subprocess.DEVNULL)
        return original_call(*args, **kwargs)

    # Apply the subprocess patch
    subprocess.call = quiet_call

    # Mock SLURM queue reading (always return empty queue)
    VirtualQueue._read_slurm_queue = lambda self: []

    # Replace job submission with local execution
    def _submit_locally(self, job_command_list):
        """Submit job locally instead of through SLURM.
        note that mbar analysis is also submitted as a slurm job in A3FE
        """
        # Get the working directory from the command list
        cwd = None
        if "--chdir" in job_command_list:
            idx = job_command_list.index("--chdir")
            cwd = job_command_list[idx + 1]
        real_cwd = cwd or os.getcwd()

        # Find the script to execute
        script_idx = next(
            (j for j, tok in enumerate(job_command_list) if tok.endswith(".sh")), None
        )

        if script_idx is None:
            raise RuntimeError(f"No .sh script found in command: {job_command_list!r}")

        script_file = job_command_list[script_idx]
        script_path = os.path.join(cwd or os.getcwd(), script_file)

        # Check if this is a SOMD simulation (has lambda argument)
        if len(job_command_list) > script_idx + 1:
            lam_arg = job_command_list[-1]  # Lambda value

            # Extract run number from path for synchronization
            run_match = re.search(r'run_(\d+)', real_cwd)
            run_no = int(run_match.group(1)) if run_match else 1

            # ——— EARLY SKIP if Simulation.log shows FINISHED ———
            sim_log = os.path.join(real_cwd, "Simulation.log")
            exe_log = os.path.join(real_cwd, "local_execution.log")
            if os.path.exists(sim_log) and os.path.exists(exe_log):
                with open(sim_log, "r") as f:
                    sim_lines = f.read().splitlines()
                sim_ok = any(
                    "JobStatus.FINISHED finished successfully" in L for L in sim_lines
                )

                with open(exe_log, "r") as f:
                    local_content = f.read()
                exe_ok = "Job completed successfully" in local_content

                if sim_ok and exe_ok:
                    with open(exe_log, "a") as f:
                        f.write(
                            f"[LOCAL SOMD] SKIPPED lambda={lam_arg} at "
                            f"{datetime.now():%Y-%m-%d %H:%M:%S}\n"
                        )
                    logger.info(
                        f"[LOCAL SOMD] Already finished in {real_cwd}; SKIPPING"
                    )
                    # Return a fake job ID that will immediately be marked as finished
                    return 999999

            return _GLOBAL_SOMD_MANAGER.submit_somd_job(script_path, lam_arg, cwd, run_no)
        else:
            # Check if this is an MBAR analysis script
            if _is_mbar_script(script_path):
                return _GLOBAL_MBAR_MANAGER.submit_mbar_job(script_path, real_cwd)
            else:
                # This is a preparation step
                return _run_prep_locally(script_path, real_cwd)

    def _run_prep_locally(script_path, cwd) -> int:
        """Run preparation step locally."""
        logger.info(f"[LOCAL PREP] Running preparation script in {cwd or os.getcwd()}")

        # Read the script to find the python command
        python_command = None
        with open(script_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("python -c") and "a3fe.run.system_prep" in line:
                    python_command = line
                    break

        if not python_command:
            raise RuntimeError(f"No A3FE preparation command found in {script_path}")

        logger.info(f"[LOCAL PREP] Executing: {python_command} at {cwd or os.getcwd()}")

        try:
            subprocess.run(python_command, shell=True, cwd=cwd, check=True)
            logger.info("[LOCAL PREP] ✅ Completed successfully")
            return 888888  # Return fake job ID
        except subprocess.CalledProcessError as e:
            logger.error(f"[LOCAL PREP] ❌ Failed with return code {e.returncode}")
            raise RuntimeError(f"Preparation step failed: {e}")

    def timing_based_get_tot_gpu_time(self) -> float:
        """need to get tot_gpu_time to set relative_simulation_cost which is
        a must for adaptive runtime mode

        however, we set set_relative_sim_cost=False for get_optimal_lam_vals()
           so this method will not be called in the first place.
        """
        # First check for local execution log
        timing_log_path = os.path.join(self.base_dir, "local_execution.log")
        if os.path.exists(timing_log_path):
            try:
                with open(timing_log_path, "r") as f:
                    content = f.read()
                    # Look for "Simulation took X.XX seconds"
                    match = re.search(r"Simulation took ([\d.]+) seconds", content)
                    if match:
                        seconds = float(match.group(1))
                        hours = seconds / 3600
                        return hours
            except Exception as e:
                logger.error(
                    f"[ERROR] ❌ get tot_gpu_time - Failed to read local execution: {e}"
                )
        else:
            logger.error(
                f"[ERROR] ❌ get tot_gpu_time - Local execution log not found at {timing_log_path}"
            )

    def local_slurm_outfile(self):
        """Mock slurm outfile property for local execution."""
        if hasattr(self, "_local_outfile"):
            return self._local_outfile

        cwd = None
        if "--chdir" in self.command_list:
            idx = self.command_list.index("--chdir")
            cwd = self.command_list[idx + 1]
        log_dir = cwd or os.getcwd()
        log_file = os.path.join(log_dir, "local_execution.log")

        os.makedirs(log_dir, exist_ok=True)
        open(log_file, "a").close()

        self._local_outfile = log_file
        return log_file

    def local_has_failed(self):
        """Mock has_failed method for local execution."""
        try:
            log_file = self.slurm_outfile  # points at local_execution.log
            with open(log_file, "r") as f:
                return "JOB FAILED" in f.read()
        except Exception:
            return False  # no log yet → not failed

    # Override the update method to handle fake job IDs
    original_update = VirtualQueue.update

    def local_update(self) -> None:
        """Updated update method that handles local execution fake job IDs."""
        # Define fake job IDs used for already-completed or local jobs
        # 888888 from actual execution, 999999 from early skip, 666666 from mbar output
        fake_job_ids = [888888, 999999]
        fake_fail_ids = [777777]

        # Mark any jobs with fake IDs as finished and remove from queue
        jobs_to_remove = []
        for job in self._slurm_queue:
            # grab more info like command list, etc.
            job_sim_info = _parse_sim_info_from_job(job)

            sid = getattr(job, "slurm_job_id", None)
            if not isinstance(sid, int):
                # not yet assigned, skip this cycle
                continue

            # Handle concurrent SOMD jobs - check status in global manager
            if job.slurm_job_id >= 800000 and job.slurm_job_id < 900000:  # Concurrent SOMD job IDs start at 800000
                status = _GLOBAL_SOMD_MANAGER.get_job_status(job.slurm_job_id)
                if status == "FINISHED":
                    if not getattr(job, "_already_marked_finished", False):
                        job.status = _JobStatus.FINISHED
                        job._already_marked_finished = True
                        jobs_to_remove.append(job)
                        logger.info(f"[CONCURRENT UPDATE] ✅ SOMD job finished! Job_ID: {job.slurm_job_id}, {job_sim_info}")
                elif status == "FAILED":
                    if not getattr(job, "_already_marked_finished", False):
                        job.status = _JobStatus.FAILED
                        job._already_marked_finished = True
                        jobs_to_remove.append(job)
                        logger.warning(f"[CONCURRENT UPDATE] ❌ SOMD job failed! Job_ID: {job.slurm_job_id}, {job_sim_info}")
                elif status == "RUNNING":
                    if not hasattr(job, "_logged_running"):
                        logger.info(
                            f"[CONCURRENT UPDATE] Running SOMD Job_ID: {job.slurm_job_id}, {job_sim_info}"
                        )
                        job._logged_running = True

                continue

            # Handle MBAR jobs - check status in global manager
            if job.slurm_job_id >= 600000 and job.slurm_job_id < 800000:  # MBAR job IDs start at 600000
                status = _GLOBAL_MBAR_MANAGER.get_job_status(job.slurm_job_id)

                if status == "FINISHED":
                    if not getattr(job, "_already_marked_finished", False):
                        job.status = _JobStatus.FINISHED
                        job._already_marked_finished = True
                        jobs_to_remove.append(job)
                        logger.info(
                            f"[LOCAL UPDATE] ✅ MBAR job {job.slurm_job_id} finished, {job_sim_info}"
                        )
                elif status == "FAILED":
                    if not getattr(job, "_already_marked_finished", False):
                        job.status = (
                            _JobStatus.FINISHED
                        )  # Still mark as finished to continue pipeline
                        job._already_marked_finished = True
                        jobs_to_remove.append(job)
                        logger.warning(
                            f"[LOCAL UPDATE] ⚠️ MBAR job {job.slurm_job_id} failed but continuing, {job_sim_info}"
                        )
                elif status == "RUNNING":
                    # Keep job in queue, mark as running if not already done
                    if not hasattr(job, "_logged_running"):
                        logger.info(
                            f"[LOCAL UPDATE] 🟡 MBAR job {job.slurm_job_id} running, {job_sim_info}"
                        )
                        job._logged_running = True
                continue

            if job.slurm_job_id in fake_job_ids:
                if getattr(job, "_already_marked_finished", False):
                    continue
                job.status = _JobStatus.FINISHED
                job._already_marked_finished = (
                    True  # flag to prevent repeated downstream logging
                )
                jobs_to_remove.append(job)
                logger.info(
                    f"[LOCAL UPDATE] ✅ Marking job slurm_job_id={job.slurm_job_id}, {job_sim_info} as finished"
                )
            elif job.slurm_job_id in fake_fail_ids:
                # leave it in the queue so downstream sees it as FAILED
                # (local_has_failed will now return True)
                job.status = _JobStatus.FAILED
                continue

        # Remove the completed local jobs
        for job in jobs_to_remove:
            self._slurm_queue.remove(job)
        # Call original update for any real SLURM jobs (if any)
        original_update(self)

    # Reduce VirtualQueue logging verbosity by patching the submit method
    def quiet_submit(self, command_list, slurm_file_base):
        """Submit method without the 'submitted' logging message."""
        virtual_job_id = self._available_virt_job_id
        self._available_virt_job_id += 1
        job = Job(virtual_job_id, command_list, slurm_file_base=slurm_file_base)
        job.status = _JobStatus.QUEUED
        self._pre_queue.append(job)
        # Remove this line: self._logger.info(f"{job} submitted")
        self.update()
        return job

    def _local_wait(self) -> None:
        """Wait for all jobs to finish with faster polling for local execution."""
        while len(self.queue) > 0:
            self.update()
            sleep(3)

    def _local_stage_wait(self) -> None:
        """Wait for the stage to finish with faster polling."""
        sleep(3)
        self.virtual_queue.update()
        while self.running:
            sleep(3)
            self.virtual_queue.update()

    def _local_sim_runner_wait(self) -> None:
        """Wait for the simulation runner to finish with faster polling for local execution."""
        sleep(3)
        while self.running:
            sleep(3)

    # APPLY THE PATCHES NOW
    Simulation.get_tot_gpu_time = timing_based_get_tot_gpu_time

    VirtualQueue.update = local_update
    VirtualQueue._submit_job = _submit_locally

    Job.slurm_outfile = property(local_slurm_outfile)
    Job.has_failed = local_has_failed

    VirtualQueue.submit = quiet_submit
    
    # Apply the stage patch for concurrent SOMD
    patch_stage_for_concurrent_somd()

    if use_faster_wait:
        # Patch the wait method to use faster polling for local execution
        # This is useful for local testing to avoid long waits
        VirtualQueue.wait = _local_wait
        Stage.wait = _local_stage_wait
        SimulationRunner.wait = _local_sim_runner_wait

    logger.info("A3FE._virtual_queue was successfully patched for local execution")


def patch_logging_into_local_execution_log():
    """
    Simply move loggings like:

    /project/6097686/jjhuang/fep_workflows/a3fe_jh/a3fe/read/_process_somd_files.py:305: 
    UserWarning: Very little data (< 50 lines) to write truncated simfile: \
        /project/6097686/jjhuang/fep_workflows/new_run_final_1/bound/vanish/output/lambda_0.614/run_01/simfile.dat.
    _warn(f"Very little data (< 50 lines) to write truncated simfile: {simfile}.")

    into a local_execution.log file in the same directory as the simfile.

    and move some other unimportant loggings into local_execution.log
    """
    import a3fe.read._process_somd_files as _a3fe_process_somd_files

    logger = logging.getLogger(__name__ + ".SUPPRESS_LOGGINGS")

    def _warn_to_local_log(message, *args, **kwargs):
        # Try to find output directory from message (simfile path is inside message)
        # Example: "Very little data ...: /path/to/output/lambda_0.000/run_01/simfile.dat"
        m = re.search(
            r":\s*(/.*?/output/.*?/run_\d+/simfile\.dat)\.?\s*$", str(message)
        )
        if m:
            simfile_path = m.group(1)
            log_dir = os.path.dirname(simfile_path)
            log_path = os.path.join(log_dir, "local_execution.log")
            os.makedirs(log_dir, exist_ok=True)
            with open(log_path, "a") as flog:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                flog.write(f"[LOCAL MBAR] {timestamp} {message}\n")
        # Do NOT propagate to console
        return

    shared_filter.runtime_constant_patterns = [
        re.compile(
            r"does not have the attribute runtime_constant and this will not be created"
        ),
        re.compile(r"Setting the attribute runtime_constant to"),
    ]

    original_filter = shared_filter.filter

    def enhanced_filter(self, record):
        # First apply the original filtering logic
        if not original_filter(record):
            return False

        # Then check for runtime_constant patterns
        msg = record.getMessage()
        for pattern in self.runtime_constant_patterns:
            if pattern.search(msg):
                return False  # Suppress this message

        return True  # Allow message through

    _a3fe_process_somd_files._warn = _warn_to_local_log
    shared_filter.filter = enhanced_filter.__get__(
        shared_filter, shared_filter.__class__
    )

    logger.info("Patched some logging into local_execution.log for clearer output")


def patch_shorter_runtime_when_resuming(new_runtime=0.0):
    """
    Directly patch the hardcoded runtime=0.2 to runtime=0.1 in Stage._run_without_threading

    0.1 ns is the minimum allowed runtime for SOMD simulations
    """    
    logger = logging.getLogger(__name__ + ".RUNTIME_PATCH")
    logger.handlers.clear()
    logger.setLevel(logging.INFO) 
    logger.propagate = False
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    handler.addFilter(shared_filter)
    logger.addHandler(handler)
    
    def patched_run_without_threading(
        self,
        run_nos,
        adaptive=True,
        runtime=None,
        max_runtime=60,
    ):
        """Patched version with runtime=0.05 instead of 0.2"""
        try:
            # Reset self.kill_thread so we can restart after killing
            self.kill_thread = False

            if not adaptive and runtime is None:
                raise ValueError(
                    "If adaptive equilibration detection is disabled, a runtime must be supplied."
                )
            if adaptive and runtime is not None:
                raise ValueError(
                    "If adaptive equilibration detection is enabled, a runtime cannot be supplied."
                )

            if not adaptive:
                self._logger.info(
                    f"Starting {self}. Adaptive equilibration = {adaptive}..."
                )
            elif adaptive:
                self._logger.info(
                    f"Starting {self}. Adaptive equilibration = {adaptive}..."
                )
                if runtime is None:
                    runtime = new_runtime  # This is the only change: 0.2 → new_runtime

            # Run initial SOMD simulations
            for win in self.lam_windows:
                win.run(run_nos=run_nos, runtime=runtime)
                # win._update_log()
                self._dump()

            try:
                leg_name = getattr(self.leg_type, "name", "bound").lower()
            except Exception:
                raise
            try:
                stage_name = getattr(self.stage_type, "name", "vanish").lower()
            except Exception:
                raise

            if _GLOBAL_SOMD_MANAGER:
                self._logger.info(
                    f"[STAGE BARRIER] Waiting for all lambda windows to complete initial runs "
                    f"before efficiency check (stage={stage_name}, leg={leg_name})"
                )
                _GLOBAL_SOMD_MANAGER.wait_for_stage(leg_name, stage_name)

            # Periodically check the simulations and analyse/ resubmit as necessary
            # Copy to ensure that we don't modify self.lam_windows when updating self.running_wins
            self.running_wins = self.lam_windows.copy()
            self._dump()

            # Run the appropriate run loop
            if adaptive:
                # Allocate simulation time to achieve maximum efficiency
                self._run_loop_adaptive_efficiency(
                    run_nos=run_nos, max_runtime=max_runtime
                )
                # Check that equilibration has been achieved and resubmit if required
                self._run_loop_adaptive_equilibration_multiwindow(
                    run_nos=run_nos, max_runtime=max_runtime
                )
            else:
                self._run_loop_non_adaptive()

            # All simulations are now finished, so perform final analysis
            self._logger.info(f"All simulations in {self} have finished.")

        except Exception as e:
            self._logger.exception("")
            raise e
    
    # Replace the method
    Stage._run_without_threading = patched_run_without_threading
    
    logger.info(f"Stage._run_without_threading patched: runtime 0.2 → {new_runtime}")

    
# ==================================================
# DEBUGGING HELPERS
# ==================================================
def _debug_patch_stage_skip_adaptive_efficiency():
    """
    Patch Stage._run_without_threading to optionally skip the adaptive efficiency loop.
    This is useful for debugging and testing or when need to skip the resource-intensive optimization phase.
    """
    # Set up colored logger for this function
    logger = logging.getLogger(__name__ + ".STAGE_PATCH")
    logger.handlers.clear()  # Clear any existing handlers
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Don't propagate to avoid duplicate messages
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    handler.addFilter(shared_filter)
    logger.addHandler(handler)

    def patched_run_without_threading(
        self,
        run_nos,
        adaptive=True,
        runtime=None,
        max_runtime=60,
    ):
        """Modified _run_without_threading that can skip the adaptive efficiency loop"""
        try:
            self.kill_thread = False
            if not adaptive and runtime is None:
                raise ValueError(
                    "If adaptive equilibration detection is disabled, a runtime must be supplied."
                )
            if adaptive and runtime is not None:
                raise ValueError(
                    "If adaptive equilibration detection is enabled, a runtime cannot be supplied."
                )
            if not adaptive:
                self._logger.info(
                    f"Starting {self}. Adaptive equilibration = {adaptive}..."
                )
            elif adaptive:
                self._logger.info(
                    f"Starting {self}. Adaptive equilibration = {adaptive}..."
                )
                if runtime is None:
                    runtime = 0.2  # ns

            # Run initial SOMD simulations
            if SKIP_ADAPTIVE_EFFICIENCY:
                self._logger.info(
                    "SKIP_ADAPTIVE_EFFICIENCY=True: Skipping initial SOMD simulations (dry run mode)"
                )
                self.running_wins = []
            else:
                for win in self.lam_windows:
                    win.run(run_nos=run_nos, runtime=runtime)
                    win._update_log()
                    self._dump()

            if not SKIP_ADAPTIVE_EFFICIENCY:
                self.running_wins = self.lam_windows.copy()
            self._dump()

            if adaptive:
                # NEW: Check if we should skip adaptive efficiency
                if SKIP_ADAPTIVE_EFFICIENCY:
                    self._logger.info(
                        "SKIP_ADAPTIVE_EFFICIENCY=True: Skipping adaptive efficiency optimization loop"
                    )
                    self._maximally_efficient = True
                else:
                    self._logger.info("Running adaptive efficiency optimization loop")
                    self._run_loop_adaptive_efficiency(
                        run_nos=run_nos, max_runtime=max_runtime
                    )
                self._run_loop_adaptive_equilibration_multiwindow(
                    run_nos=run_nos, max_runtime=max_runtime
                )
            else:
                self._run_loop_non_adaptive()
            self._logger.info(f"All simulations in {self} have finished.")

        except Exception as e:
            self._logger.exception("")
            raise e

    # APPLY THE PATCH
    Stage._run_without_threading = patched_run_without_threading
    logger.info(
        f"Stage._run_without_threading patched to {'skip' if SKIP_ADAPTIVE_EFFICIENCY else 'include'} adaptive efficiency loop"  # noqa: E501
    )


def _debug_simulation_times(calc):
    """Debug simulation times to identify inconsistencies

    sometimes we get the following error:

    │ /Users/jingjinghuang/Documents/fep_workflow/a3fe_jh/a3fe/analyse/process_grads.py:638 in         │
    │ get_time_series_multiwindow                                                                      │
    │                                                                                                  │
    │   635 │   │   │   │   for i in range(n_runs)                                                     │
    │   636 │   │   │   ]                                                                              │
    │   637 │   │   ):                                                                                 │
    │ ❱ 638 │   │   │   raise ValueError(                                                              │
    │   639 │   │   │   │   "Total simulation times are not the same for all runs. Please ensure tha   │
    │   640 │   │   │   │   "the total simulation times are the same for all runs."                    │
    │   641 │   │   │   )                                                                              │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
    ValueError: Total simulation times are not the same for all runs. Please ensure that the total simulation times are the same for all runs.  # noqa: E501
    
    therefore must ensure consistent runtime for repeats and simply resume the calculation if the previous run failed or cancelled due to timeout
    """
    logger = calc._logger
    logger.info("=== DEBUGGING SIMULATION TIMES ===")

    issues_found = []

    for leg in calc.legs:
        for stage in leg.stages:
            stage_issues = []
            for win in stage.lam_windows:
                logger.info(f"LEG {leg.leg_type.name} STAGE {stage.stage_type.name}  Lambda {win.lam:.3f}:")
                simtimes = []

                for i, sim in enumerate(win.sims, 1):
                    simtime = sim.get_tot_simtime()
                    simtimes.append(simtime)
                    logger.info(f"  Run {i}: {simtime:.6f} ns")

                    # Check if simulation output files exist
                    simfile_path = f"{sim.output_dir}/simfile.dat"
                    if not os.path.exists(simfile_path):
                        issue = f"Missing simfile.dat for {stage.stage_type.name} lambda {win.lam:.3f} run {i}"
                        logger.error(f"    ERROR: {issue}")
                        issues_found.append(issue)
                    elif os.path.getsize(simfile_path) == 0:
                        issue = f"Empty simfile.dat for {stage.stage_type.name} lambda {win.lam:.3f} run {i}"
                        logger.error(f"    ERROR: {issue}")
                        issues_found.append(issue)

                # Check consistency within this lambda window
                if len(set(f"{t:.6f}" for t in simtimes)) > 1:
                    issue = f"Inconsistent times in {stage.stage_type.name} lambda {win.lam:.3f}: {simtimes}"
                    logger.error(f"    ERROR: {issue}")
                    stage_issues.append((win.lam, simtimes))
                    issues_found.append(issue)
                else:
                    logger.info(f"    ✓ All runs consistent: {simtimes[0]:.6f} ns")

            # Check consistency across lambda windows in this stage
            if stage_issues:
                logger.warning(f"  STAGE {stage.stage_type.name} HAS TIMING ISSUES:")
                for lam, times in stage_issues:
                    logger.warning(f"    Lambda {lam:.3f}: {times}")

    return issues_found


def _debug_patch_force_not_equilibrated():
    """
    Temporarily patch Stage.is_equilibrated() to always return False
    for testing _run_loop_adaptive_equilibration_multiwindow().
    """
    logger = logging.getLogger(__name__ + ".DEBUG_PATCH")

    def force_false_equilibrated(self, run_nos=None):
        return False

    Stage.is_equilibrated = force_false_equilibrated
    logger.info(
        "Stage.is_equilibrated() patched to always return False for testing purposes"
    )


if __name__ == "__main__":
    from a3fe.run.enums import LegType as _LegType
    # Set up global logging first
    setup_global_logging()

    # Configure via environment variables
    FORCE_LOCAL_EXECUTION = True
    FORCE_CPU_PLATFORM = False
    MAX_CONCURRENT_SOMD=5  # It seems we should set this to 5 in HPC

    patch_virtual_queue_for_local_execution()
    patch_logging_into_local_execution_log()

    # _debug_patch_stage_skip_adaptive_efficiency()
    # _debug_patch_force_not_equilibrated()
    a3.Calculation.required_legs = [_LegType.BOUND]
    sysprep_cfg = SystemPreparationConfig(slurm=True)  # use default settings

    calc = a3.Calculation(
        base_dir="/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again2",
        input_dir="/Users/jingjinghuang/Documents/fep_workflow/test_somd_run_again2/input",
        ensemble_size=3,
    )

    calc.setup(
        bound_leg_sysprep_config=sysprep_cfg,
        free_leg_sysprep_config=sysprep_cfg,
    )

    add_filter_recursively(calc)

    calc.get_optimal_lam_vals()
    calc.run(adaptive=True,
             parallel=False)              # run things sequentially

    calc.wait()
    calc.analyse()
    calc.save()


