#!/bin/bash
#SBATCH --partition=cpu                # In Vega should bw cpu, long, or gpu 
#SBATCH --time=10:00:00                     # Strictly recommended when using 'general' partition.
#SBATCH --output=out_err/run_info_eu2_mv.out              # Generate custom output file
#SBATCH --error=out_err/run_info_eu2_mv.err                # Generate custom error  file
#SBATCH --ntasks-per-core=1                 # Recommended one thread per core
   ##SBATCH --exclusive                        # Uncomment if you need exclusive node usage

#SBATCH --hint=nomultit
#SBATCH --nodes=1                          # Uncomment and specify #nodes to use
#SBATCH --ntasks=2                        # Uncomment and specify #nodes to use  
#SBATCH --ntasks-per-node=2               # Uncomment and specify #tasks per node
##SBATCH --job-name=testjim
##SBATCH --mail-type=END
##SBATCH --mail-user=marta.viagonzalez@ung.si

#echo task begin

srun python experiment_runner.py --procs 2 --input json/bamf_tutorial.json --log results/run_info_eu2.h5
##srun    python experiment_runner.py --procs 2 --input json/bamf_tutorial_normal_mode.json --log results/run_info_normal.h5
