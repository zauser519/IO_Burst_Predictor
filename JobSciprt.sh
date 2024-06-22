#!/bin/sh
#PBS -q sxs
#PBS --venode 1
#PBS -l elapstim_req=03:00:00

cd /uhome/a01431/Sample
export OMP_NUM_THREADS=32
/uhome/a01431/miniconda3/bin/python C.py
