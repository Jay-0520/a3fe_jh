a3fe
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/michellab/a3fe/workflows/CI/badge.svg)](https://github.com/fjclark/a3fe/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/michellab/a3fe/graph/badge.svg?token=5IGO8SCRRQ)](https://codecov.io/gh/michellab/a3fe)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation Status](https://readthedocs.org/projects/a3fe/badge/?version=latest)](https://a3fe.readthedocs.io/en/latest/?badge=latest)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
      
<img src="./a3fe_logo.png" alt="Alt text" style="width: 50%; height: 50%;">

**A**utomated **A**daptive **A**bsolute alchemical **F**ree **E**nergy calculator. A package for running adaptive alchemical absolute binding free energy calculations with SOMD (distributed within [sire](https://sire.openbiosim.org/)) using adaptive protocols based on an ensemble of simulations. This requires the SLURM scheduling system. Please see the [**documentation**](https://a3fe.readthedocs.io/en/latest/?badge=latest).

For details of the algorithms and testing, please see the assocated paper:

**Clark, F.; Robb, G. R.; Cole, D. J.; Michel, J. Automated Adaptive Absolute Binding Free Energy Calculations. J. Chem. Theory Comput. 2024, 20 (18), 7806–7828. https://doi.org/10.1021/acs.jctc.4c00806.**

### Installation

a3fe depends on SLURM for scheduling jobs, and on GROMACS for running initial equilibration simulations. Please ensure that your have sourced your GMXRC or loaded your GROMACS module before proceeding with the installation. While we recommend installing with [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html), you can substitute `mamba` with `conda` in the following commands.

Now, download and install a3fe:
```bash
git clone https://github.com/michellab/a3fe.git
cd a3fe
mamba env create -f environment.yaml
mamba activate a3fe
python -m pip install --no-deps .
```

### How to run on different clusters with GPU (DRAC | Compute Canada)
- According to [creating and using a virtual environment](https://docs.alliancecan.ca/wiki/Python#Creating_and_using_a_virtual_environment), it's strongly recommended against using Conda or Anaconda in DRAC HPC. This will cause issues with setting CUDA because Conda or Anaconda will pull in its own compilers, CUDA runtimes, and library paths, and mess up with Modules in DRAC HPC cluster. This is also why we need to use `export LD_LIBRARY_PATH="$CUDA_HOME/lib64"` on Graham to get around this issue. However, this quick fix is not recommended, so better to use `mamba env create -f environment.yaml`!
- On Graham, we should use the following environment.yaml file: 
    ```
    name: a3fe_gra
    channels:
    - conda-forge
    - openbiosim

    dependencies:
        # Base depends
    - arviz
    - pandas
    - python>=3.9
    - pip
    - openmm<8.3.0
    - numpy
    - matplotlib
    - scipy
    - ipython
    - pymbar<4
    - ambertools
    - biosimspace<=2024.4.1
    - pydantic
    - loguru
    ```
  we should add these modules/headers into sbatch script for Graham:
    ```
    module --force purge
    module load StdEnv/2020  gcc/9.3.0  cuda/11.8.0  openmpi/4.0.3
    module load gromacs/2023

    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate a3fe_gra

    unset LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
    ```
- it's better to set MEM >= 16G in slurm script to avoid OOM errors


### Running a3fe in one node
- It is highly recommended to use script `a3fe_jh/a3fe/data/example_run_jh/script_run_in_1node_enhanced.py` ([link](https://github.com/Jay-0520/a3fe_jh/blob/main/a3fe/data/example_run_jh/script_run_in_1node_enhanced.py)) to run a3fe calculation in one node, especially in a HPC environment, since queueing can be very slow. 
  - for running in adaptive mode, make sure to remove this step `calc.set_equilibration_time()`


### Quick start for simple test
- Activate your a3fe conda environment 
- Create a base directory for the calculation and create an directory called `input` within this
- Move your input files into the the input directory. For example, if you have parameterised AMBER-format input files, name these bound_param.rst7, bound_param.prm7, free_param.rst7, and free_param.prm7. For more details see the documentation. Alternatively, copy the example input files from a3fe/a3fe/data/example_run_dir to your input directory.
- Copy run somd.sh and template_config.sh from a3fe/a3fe/data/example_run_dir to your `input` directory, making sure to the SLURM options in run_somd.sh so that the jobs will run on your cluster
- In the calculation base directory, run the following python code, either through ipython or as a python script (you will likely want to run the script with `nohup`or use ipython through tmux to ensure that the calculation is not killed when you lose connection)
- a up-to-date run script can be found here: `a3fe_jh/a3fe/data/example_run_jh/script_run_in_1node_enhanced.py` (see below how to run things in one node)
    - in the same folder, we can use `protein.pdb` and `ligand.sdf` as the input to quickly test the run
- Check the results in the ``output`` directories (separate output directories are created for the Calculation, Legs, and Stages)

```python
import a3fe as a3

# we can change slurm script setting here for preparation steps
for step in ["parameterise", "solvate", "minimise", "heat_preequil", "ensemble_equil"]:
    a3.Leg.update_default_slurm_config(
        step_type=step,
        time="12:00:00",
        gres="",  # switch to CPU-only 
        pre_commands=['export PATH="$CONDA_PREFIX/bin:$PATH"']
    )

calc = a3.Calculation(
    ensemble_size=3,
    base_dir="~/test_run_full/",
    input_dir="~/test_run_full/input",
)
calc.setup()

# we could also update somd slurm script here
calc.bound_leg.update_slurm_script(
    "somd_production",
    mem="2G",             
    time="00:12:12",
    gres="", # switch to CPU-only
    setup_cuda_env=False,       # Disable CUDA environment setup
    somd_platform="CPU"         # Use CPU instead of CUDA 
)

calc.get_optimal_lam_vals()
calc.run(adaptive=False, runtime = 5) # Run non-adaptively for 5 ns per replicate
calc.wait()
calc.set_equilibration_time(1) # Discard the first ns of simulation time
calc.analyse()
calc.save()
```

### Some notes for solving runtime errors 
 (more runtime errors can be found in this [README.md](https://github.com/Jay-0520/a3fe_jh/blob/main/a3fe/data/example_run_jh/README.md))
- if `ensemble_equilibration_*` runs faild, we can simply remove the entire folder and re-run the calculation
- reloading molecules via `_BSS.IO.readMolecules()` leads to different molecule number (MolNum). This may cause issues when we resume a previously-stopped run
  - as a result, when using `skip_preparation=True` in Leg.setup(), we have to ensure that restraints are generated in the same run as  pre-equilibrated system is loaded. In other words, `restraints.pkl` and `Leg.pkl` must be created in the same run
- pay attention to calculation.pkl (leg.pkl or stage.pkl) files when re-running a previously stopped calculation because the calculation will load these pickle file by default. These pickle files are use to load the previously saved `Calculation`, `Leg` and `Stage` objects.
- some `simfile.dat` files generated from SOMD simulations may miss headers.
- we might get this error at runtime
    ```
    │ /home/jjhuang/miniconda3/envs/a3fe_gra/lib/python3.12/site-packages/BioSimSp │
    │ ace/Sandpit/Exscientia/Parameters/_parameters.py:568 in _parameterise_openff │
    │                                                                              │
    │    565 │   │   string = string.replace("Welcome to antechamber", "")         │
    │    566 │   │                                                                 │
    │    567 │   │   # Extract the version and convert to float.                   │
    │ ❱  568 │   │   version = float(string.split(":")[0])                         │
    │    569 │   │                                                                 │
    │    570 │   │   # The version is okay, enable Open Force Field support.       │
    │    571 │   │   if version >= 22:                                             │
    ╰──────────────────────────────────────────────────────────────────────────────╯
    ```
    this is simply because calling `antechamber -v` failed to produce right version info. we could install a different version
    or simply set `version=23`.
- when we get this error at runtime (see below), we should double check and downgrade `openff` package versions. First init your conda env, and then try this installation command `conda install -c conda-forge openff-toolkit=0.16.9 openff-interchange=0.4.3`. 
    ```
    │ /home/jjhuang/miniconda3/envs/a3fe_jh/lib/python3.12/site-packages/BioSimSpa │
    │ ce/Sandpit/Exscientia/Parameters/_process.py:244 in getMolecule              │
    │                                                                              │
    │   241 │   │   │   │   │   "Parameterisation failed! Last error: '%s'" % str( │
    │   242 │   │   │   │   )                                                      │
    │   243 │   │   │   else:                                                      │
    │ ❱ 244 │   │   │   │   raise _ParameterisationError(                          │
    │   245 │   │   │   │   │   "Parameterisation failed! Last error: '%s'" % str( │
    │   246 │   │   │   │   ) from None                                            │
    │   247                                                                        │
    ╰──────────────────────────────────────────────────────────────────────────────╯
    ParameterisationError: Parameterisation failed! Last error: 'Unable to create 
    OpenFF Interchange object!'
    See this [issue](https://github.com/openforcefield/openff-interchange/issues/1309) for more details 
    ```    

### How to resume previously-killed run? 
- we often need to resume calculation to extend simulations due to the sampling challenge of FEP. we can use this tool `a3fe.run.fix_simulation_times.py` to check simulations runs that were interrupted. we must ensure all runs for a given lambda have completed the same amount of runtime.
  - first of all, use `fix_simulation_times(calc, dry_run=False)` to remove data files for all runs of the lambda window with inconsistent runtime
    - This is because SOMD simulation `always` (need to double check this) restart from previous check point if a .s3 file exist. Without a .s3 file,
    simulation will always start from the begining any way
  - secondly, we can apply `patch_shorter_runtime_when_resuming(new_runtime=0.1)` from `script_run_in_1node_enhanced.py` avoid running additional simulations for lambdaw windows that were already completed:
    ```
        calc = a3.Calculation(base_dir="previous_one", input_dir="previous_one")
        # this patch can be useful to speed up the process
        # 0.1 is the minimum in a3fe 
        patch_shorter_runtime_when_resuming(0.1) 
        calc.run(adaptive=True,
                parallel=True)
        calc.wait()
        calc.analyse()
        calc.save()
    ```
    - Note that `new_runtime` must be set to 0.1 which is the minimum because when resuming a calculation, we very likely clean up some runs because of the inconsistency runtime (see step 1)
  - Finally just for noting, I have to re-align the time-axis in the `get_time_series_multiwindow()` and `get_time_series_multiwindow_mbar()` because the original code raise an error if runtimes are not consistency between runs (sum of all lambda windows). The monkey patch I introduced simply get the minimum runtime of the runs and re-align the time-axis. This shouldn't affect any physics in the pipeline.   
    

 
### Copyright

Copyright (c) 2023, Finlay Clark


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
