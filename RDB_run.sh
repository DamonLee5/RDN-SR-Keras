#!/bin/bash
module use /depot/itap/amaji/modules
module spider learning
module load learning/conda-5.1.0-py36-gpu
module load ml-toolkit-gpu/tensorflow
module load ml-toolkit-gpu/keras
module load  ml-toolkit-gpu/opencv
module list
source activate /home/li3120/.conda/envs/cent7/5.1.0-py36/tfgan1
cd /home/li3120/ECE570
python /home/li3120/ECE570/model.py
