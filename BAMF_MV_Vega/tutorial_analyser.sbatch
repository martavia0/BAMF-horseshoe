#!/bin/bash
#SBATCH --partition=cpu                 # Specify 'general' or 'daily' or 'hourly'
#SBATCH --time=10:00:00                     # Strictly recommended when using 'general' partition.
#SBATCH --output=out_err/run_analy_eu1.out              # Generate custom output file
#SBATCH --error=out_err/run_analy_eu1.err                # Generate custom error  file
#SBATCH --ntasks-per-core=1                 # Recommended one thread per core
   ##SBATCH --exclusive                        # Uncomment if you need exclusive node usage
#SBATCH --hint=nomultit
#SBATCH --nodes=1                          # Uncomment and specify #nodes to use
#SBATCH --ntasks=2                        # Uncomment and specify #nodes to use  
#SBATCH --ntasks-per-node=2               # Uncomment and specify #tasks per node
#SBATCH --job-name=analyser_mv
##SBATCH --mail-type=END
##SBATCH --mail-user=marta.viagonzalez@ung.si

##echo task begin

srun python experiment_analyzer.py --input ./results/run_info_eu.h5 --output ./results/analysis/metrics_eu1.h5 --results ./plots/benchmark/eu1 --benchmark True --reference_library datafile

