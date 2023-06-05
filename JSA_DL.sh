#!/bin/bash
#SBATCH -N 1 
#SBATCH -n 1
#SBATCH -p qTRDGPUH
#SBATCH --gres=gpu:V100:1
#SBATCH -c 5
#SBATCH --mem=16g
#SBATCH -t 600
#SBATCH -J DL
#SBATCH -e jobs/err%A-%a.err
#SBATCH -o jobs/out%A-%a.out
#SBATCH -A trends53c17
#SBATCH --oversubscribe 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vitkyal@gsu.edu

sleep 5s

export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/


source /data/users2/vitkyal/softwares/miniconda3/bin/activate /data/users2/vitkyal/softwares/miniconda3/envs/AA_DL

#num of nodes per layer, layers, filter size, lr, try diff initialisation like Xavier, diff pooling - max/min, batch size, es, 
#for i in 3 10 25 1000 1 #0.001 0.0001 0.00001 #$i = 0 for no fMRI k 
#do
python /data/users2/vitkyal/projects/SMLvsDL/reprex/run_DL.py --tr_smp_sizes 350 --nReps 10 --nc 2 --bs 32 --lr 0.0001 --es 500 --pp 0 --k 0 --es_va 1 --es_pat 20 --ml '../results/s1' --mt 'AlexNet3D_Dropout' --ssd 'SampleSplits_twocomplex/'  --scorename 'ResearchGroup' --nw 4 --cr 'clx'

#done
sleep 5s



