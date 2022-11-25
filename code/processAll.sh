#!/bin/bash


if [ $# -lt 2 ]
then
    echo "Apply all recognition step to all inkml files and produce the LG files"
    echo "Copyright (c) H. Mouchere, 2018"
    echo ""
    echo "Usage: processAll <input_inkml_dir> <output_lg_dir>"
    echo ""
    exit 1
fi

INDIR=$1
OUTDIR=$2

if ! [ -d $OUTDIR ] 
then
    mkdir $OUTDIR
    mkdir hyp_$OUTDIR
    mkdir seg_$OUTDIR
    mkdir symb_$OUTDIR
fi

for file in $1/*.inkml
do
    echo "Recognize: $file"
    BNAME=`basename $file .inkml`
    OUT="$OUTDIR/$BNAME.lg"
    OUT_hyp="$OUTDIR/hyp_$BNAME.lg"
    OUT_seg="$OUTDIR/seg_$BNAME.lg"
    OUT_symb="$OUTDIR/symb_$BNAME.lg"
    python3 ./code/segmenter.py -i $file -o $OUT_hyp 
    ERR=$? # test de l'erreur au cas où...
     python3 ./code/segmentSelect.py -o $OUT_seg  $file $OUT_hyp 
    ERR=$ERR || $?
     python3 ./code/symbolReco.py  -o $OUT_symb $file $OUT_seg
    ERR=$ERR || $?
     python3 ./code/selectBestSeg.py -o $OUT $OUT_symb 
    ERR=$ERR || $?
    
    if [ $ERR -ne 0 ]
    then 
        echo "erreur !" $ERR
        exit 2
    fi
done
echo "done."