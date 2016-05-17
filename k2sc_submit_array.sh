#!/bin/sh

# Directives
#PBS -N K2SC
#PBS -W group_list=yetiastro
#PBS -t 1-14
#PBS -l nodes=1,walltime=4:00:00,mem=4000mb
#PBS -M sd2706@columbia.edu 
#PBS -m n
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/astro/users/sd2706/k2/outputs/
#PBS -e localhost:/vega/astro/users/sd2706/k2/outputs/

python c5_analysis.py ../data/all_k2sc_files.lst c5_tables/c5_k2sc_output.csv

# End of script
