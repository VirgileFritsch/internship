#!/bin/bash

echo -n "Removing previous results..."
rm -rf sim1/results
rm -f sim1/results_pipeline.txt
echo "Done"

for i in $(seq 1 1 100)
do
    echo "Simulation $i"
    echo -n "    Building simulation set $i..."
    python build_simulated_data.py > /dev/null
    echo "Done"
    
    echo -n "    Blobs matching..."
    cd ..
    ./pipeline_aux.sh > /dev/null
    cd simulation
    echo "Done"
    
    echo -n "    Comparing coordinates..."
    python comp_coord_subjects.py >> sim1/results_pipeline.txt
    echo "Done"
    echo
done

python print_final_res.py
