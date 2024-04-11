#!/bin/bash
#SBATCH --partition=daily                 # Specify 'general' or 'daily' or 'hourly'
#SBATCH --time=10:00:00                     # Strictly recommended when using 'general' partition.
#SBATCH --output=run_init.out              # Generate custom output file
#SBATCH --error=run_init.err                # Generate custom error  file
#SBATCH --ntasks-per-core=1                 # Recommended one thread per core
   ##SBATCH --exclusive                        # Uncomment if you need exclusive node usage
#SBATCH --hint=nomultit
#SBATCH --nodes=1                          # Uncomment and specify #nodes to use
#SBATCH --ntasks=2                        # Uncomment and specify #nodes to use  
#SBATCH --ntasks-per-node=2               # Uncomment and specify #tasks per node
#SBATCH --job-name=testjim
#SBATCH --mail-type=END
#SBATCH --mail-user=jimeng.wu@psi.ch

echo task begin


srun python experiment_runner.py --procs 2 --input json/bamf_tutorial_model_init.json --log results/run_init.h5
