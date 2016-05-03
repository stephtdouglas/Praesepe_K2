#!/bin/sh

# Directives
#PBS -N K2SFF
#PBS -W group_list=yetiastro
#PBS -l nodes=1,walltime=24:00:00,mem=4000mb
#PBS -M sd2706@columbia.edu 
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/astro/users/sd2706/k2/outputs/
#PBS -e localhost:/vega/astro/users/sd2706/k2/outputs/

python c5_k2sff_analysis.py

# End of script
