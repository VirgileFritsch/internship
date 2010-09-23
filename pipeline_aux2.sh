SUBJECTS=( "s12069" "s12081" "s12300" "s12401" "s12431" "s12508" "s12532" "s12539" "s12562" "s12590" "s12635" "s12636" "s12898" "s12165" "s12207" "s12344" "s12352" "s12370" "s12381" "s12405" "s12414" "s12432" )

CONTRAST=( "reading-visual_z_map" "computation-sentences_z_map" "left-right_z_map" "right-left_z_map" )

for i in ${SUBJECTS[@]}
do
  echo $i
  for c in ${CONTRAST[@]}
  do
    echo $c
    echo "SUBJECT = \"$i\"" > tmp_subject.py
    echo "CONTRAST = \"$c\"" >> tmp_subject.py
    ./pipeline_blobs_fixe.sh 3 3
    ./pipeline_blobs_fixe.sh 6 6
  done
done