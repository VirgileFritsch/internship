SUBJECTS=( "s12081" "s12300" "s12401" "s12431" "s12508" "s12532" "s12539" "s12562""s12590" "s12635" "s12636" "s12898" "s12165" "s12207" "s12344" "s12352" "s12370" "s12381" "s12405" "s12414" "s12432" )

for i in ${SUBJECTS[@]}
do
  echo $i
  echo "SUBJECT = \"$i\"" > tmp_subject.py
  ./pipeline_blobs_fixe.sh 1 6
done