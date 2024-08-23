import os
import subprocess
import argparse
import time

def create_slurm_script(config_path, script_path, experiment_series_name, experiment_name):
    slurm_script_content = f"""#!/bin/bash
#SBATCH --job-name={experiment_name}
#SBATCH --output=.out/slurm/%x/%J-%N-%t.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=tb21@princeton.edu

source ~/.bashrc
conda activate inrdetect4

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python train.py -b {config_path} -t --name {experiment_name} --devices 4
"""

    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, 'w') as slurm_file:
        slurm_file.write(slurm_script_content)

def find_configs_and_generate_scripts(config_dir, delay):
    for root, _, files in os.walk(config_dir):
        for file in files:
            if file.endswith('.yaml'):
                config_path = os.path.join(root, file)
                relative_path = os.path.relpath(config_path, config_dir)
                experiment_series_name = os.path.basename(os.path.dirname(relative_path))
                experiment_name = os.path.splitext(os.path.basename(config_path))[0]
                script_path = os.path.join('scripts', 'slurm', 'train', experiment_series_name, f'{experiment_name}.slurm')
                
                create_slurm_script(config_path, script_path, experiment_series_name, experiment_name)
                print(f"Generated SLURM script: {script_path}")

                try:
                    # Submit the SLURM script
                    subprocess.run(['sbatch', script_path], check=True)
                    print(f"Submitted SLURM script: {script_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error submitting SLURM script {script_path}: {e}")

                # Delay between submissions
                time.sleep(delay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and submit SLURM scripts for all configs in specified config directory.')
    parser.add_argument('config_dir', type=str, help='The directory containing the config files.')
    parser.add_argument('--delay', type=int, default=10, help='Delay between submitting SLURM scripts (in seconds).')
    
    args = parser.parse_args()
    
    find_configs_and_generate_scripts(args.config_dir, args.delay)
