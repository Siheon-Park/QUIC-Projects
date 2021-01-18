#!/bin/sh
#PBS -N LiH_QUCC_10
#PBS -V
#PBS -q normal
#PBS -A inhouse
#PBS -l select=30:ncpus=20:mpiprocs=20:ompthreads=1
#PBS -l walltime=48:00:00
#PBS -m abe
#PBS -M snow0369@kaist.ac.kr

cd $PBS_O_WORKDIR

module purge
module load craype-mic-knl
module load python

echo "qasm Simulation with lim=10"

python $HOME/gwons_quantum/script/Q_UCCSD_LiH.py -l 10
