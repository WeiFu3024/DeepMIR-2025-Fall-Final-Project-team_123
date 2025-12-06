#!/bin/bash

# This script set up symbolic links to required datasets and result directories, since we are using the same server.

ln -s /mnt/gestalt/home/huan/ntu/DeepMIR/RapBank/ .  # Create symbolic link to RapBank dataset
ln -s /mnt/gestalt/home/WillFu/deepMIR/final/all_in_one_results . # Create symbolic link to all in one results directory
ln -s /mnt/gestalt/home/WillFu/deepMIR/final/all_in_one_results_bgm_cut . # Create symbolic link to all in one results (bgm_cut) directory