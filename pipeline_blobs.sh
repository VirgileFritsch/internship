#!/bin/bash

scripts=( "smooth_image.py" "script_first_level_localizer.py" "demo_blob_from_image.py" "structured_diffusion_smoothing.py" "script_surface_localizer.py" "script_blob_surface_aims.py" "blobs_matching0_visu.py" )

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
	PYTHONPATH_BCKUP=$PYTHONPATH
	export PYTHONPATH=$PYTHONPATH:/usr/lib64/python2.5/site-packages:/usr/lib64/python2.5/site-packages/wx-2.8-gtk2-unicode
	ipython -wthread -noconfirm_exit -c "import matplotlib as mp; mp.rcParams['backend']='WxAgg'; import pylab as pl; pl.ion(); import visu"
	export PYTHONPATH=$PYTHONPATH_BCKUP
    elif [ "${scripts[$i]}" = "blobs_matching0_visu.py" ]
    then
	PYTHONPATH_BCKUP=$PYTHONPATH
	export PYTHONPATH=$PYTHONPATH:/usr/lib64/python2.5/site-packages:/usr/lib64/python2.5/site-packages/wx-2.8-gtk2-unicode
	ipython -wthread -noconfirm_exit -c "import matplotlib as mp; mp.rcParams['backend']='WxAgg'; import pylab as pl; pl.ion(); import blobs_matching0_visu"
	export PYTHONPATH=$PYTHONPATH_BCKUP
    else
	cp ${scripts[$i]} tmp_script.py
	echo "quit()" >> tmp_script.py
        ipython -noconfirm_exit -c "run tmp_script.py"
	rm -rf tmp_script.py
    fi
done

rm -f tmp_script.pyc
echo -e '\007'
