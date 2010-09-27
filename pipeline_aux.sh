#SUBJECTS=( "s12069" "s12081" "s12300" "s12401" "s12431" "s12508" "s12532" "s12539" "s12562" "s12590" "s12635" "s12636" "s12898" "s12165" "s12207" "s12344" "s12352" "s12370" "s12381" "s12405" "s12414" "s12432" )
SUBJECTS=( "f1" "f2" "f3" "f4" "f5" "f6" "f7" "f8" "f9" "f10" "f11" "f12" "f13" "f14" "f15" "f16" "f17" "f18" "f19" "f20" "f21" "f22" )

for i in ${SUBJECTS[@]}
do
  echo $i
  echo "SUBJECT = \"$i\"" > tmp_subject.py
  python blobs_matching.py
done