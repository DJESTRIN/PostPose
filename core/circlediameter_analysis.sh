#!/bin/bash
for i in $(seq 0 1 100); do
    echo $i
    python_file=~/PostPose/core/openfield_circlediameter_analysis.py
    python $python_file \
        --root_directory C:\\Users\\listo\\Downloads\\tmt_experiment_2024_open_field\\tmt_experiment_2024_open_field\\ \
        --force \
        --percent $i
done