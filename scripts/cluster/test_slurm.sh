#!/bin/bash
#SBATCH --job-name=my_batch_job     # Name the job
#SBATCH --output=batch_job.out      # Output file
#SBATCH --error=batch_job.err       # Error file
#SBATCH --time=00:05:00             # Max time for the job (5 min)
#SBATCH --mem=1G                    # Memory request
#SBATCH --cpus-per-task=2           # Number of CPUs
#SBATCH --qos=fast                  # fast, default (or blank), bio_ai
#SBATCH --partition=gpuq            # Queue to use
#SBATCH --partition=gpuq            # Queue to use
#SBATCH --mail-type=begin           # send email when job begins
#SBATCH --mail-type=end             # send email when job ends
#SBATCH --mail-user=martigo@cshl.edu

# Commands to run
sleep 5
echo "Job completed!"
