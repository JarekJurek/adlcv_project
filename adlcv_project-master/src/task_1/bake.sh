#!/bin/bash
### --------------- job name ------------------
#BSUB -J run3

### --------------- queue name ----------------
#BSUB -q gpuv100

### --------------- GPU request ---------------
#BSUB -gpu "num=1:mode=exclusive_process"

### --------------- number of cores -----------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### --------------- CPU memory requirements ---
#BSUB -R "rusage[mem=8GB]"

### --------------- wall-clock time (max allowed is 12:00) ---------------
#BSUB -W 12:00

### --------------- output and error files ---------------
#BSUB -o bakery/run3.out
#BSUB -e bakery/run3.err

### --------------- send email notifications -------------
# BSUB -u s242911@dtu.dk
# BSUB -B
# BSUB -N

### --------------- Load environment and run Python script ---------------
source /zhome/a2/c/213547/DLCV/adlcv-ex-1/venv/bin/activate
python ./ddpm_train.py