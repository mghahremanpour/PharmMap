#!/bin/bash
#SBATCH -J EGFR_virtual_screen
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=64G

module load medsci
module load miniforge3
source ~/.bashrc
conda activate RDKit

cd $SCRATCH_DIR
cp /home/draytj01/workspace/{EGFR_test_ligands.sdf.gz,EGFR_training_ligands.sdf.gz} .
cp -r /home/draytj01/workspace/PharmMap/pharm_map .
cp /home/draytj01/workspace/PharmMap/{virtual_screen.py,EGFR_params.yml} .

python virtual_screen.py EGFR_test_ligands.sdf.gz EGFR_params.yml -t EGFR_training_ligands.sdf.gz -o EGFR --outdir EGFR_screen/ --sim --sm --sc -u -v

cp -r ./EGFR_screen /home/draytj01/workspace