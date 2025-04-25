#!/bin/bash
### --------------- job name ------------------
#BSUB -J dlcv_condi

### --------------- queue name ----------------
#BSUB -q gpuv100

### --------------- GPU request ---------------
#BSUB -gpu "num=1:mode=exclusive_process"

### --------------- number of cores -----------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### --------------- CPU memory requirements ---
#BSUB -R "rusage[mem=4GB]"

### --------------- wall-clock time (max allowed is 12:00) ---------------
#BSUB -W 12:00

### --------------- output and error files ---------------
#BSUB -o dlcv_condi_%J.out
#BSUB -e dlcv_condi_%J.err

### --------------- send email notifications -------------
#BSUB -u s242911@dtu.dk
#BSUB -B
#BSUB -N

### --------------- Load environment and run Python script ---------------
source /zhome/a2/c/213547/DLCV/adlcv-ex-1/venv/bin/activate
python ./train.py