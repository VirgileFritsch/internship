#!/bin/bash

scripts=( "smooth_image.py" "script_first_level_localizer.py" "demo_blob_from_image.py" "structured_diffusion_smoothing.py" "script_surface_localizer.py" "script_blob_surface.py" "blobs_matching0_visu.py" )

start=$1
if [ $2 = "-1" ]
then
    end=${#scripts[*]}
else
    end=$2
fi

for i in `seq $(($start - 1)) $(($end - 1))`
do
    if [ "${scripts[$i]}" = "visu.py" ]
    then
	echo 'warning'
    elif [ "${scripts[$i]}" = "blobs_matching0_visu.py" ]
    then
	echo warning
    else
	cp ${scripts[$i]} tmp_script.py
	echo "quit()" >> tmp_script.py
	/usr/bin/ipython -noconfirm_exit -wthread -c "import matplotlib as mp; mp.rcParams['backend']='WxAgg'; import pylab as pl; pl.ion(); import tmp_script"
	rm -rf tmp_script.py
    fi
done

rm -f tmp_script.pyc
