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
cpt = 0

if ! [ -d $OUTDIR ] 
then
    mkdir $OUTDIR
    mkdir "$OUTDIR/hyp"
    mkdir "$OUTDIR/seg"
    mkdir "$OUTDIR/symb"
fi
mkdir "$OUTDIR/hyp"
mkdir "$OUTDIR/seg"
mkdir "$OUTDIR/symb"
mkdir "$OUTDIR/out"



for file in $1/*.inkml
do
    if [[ $cpt -gt 670 ]];
    then
        echo "Recognize: $file"
        BNAME=`basename $file .inkml`
        OUT="$OUTDIR/out/$BNAME.lg"
        OUT_hyp="$OUTDIR/hyp/$BNAME.lg"
        OUT_seg="$OUTDIR/seg/$BNAME.lg"
        OUT_symb="$OUTDIR/symb/$BNAME.lg"
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
    fi
    cpt=$(($cpt+1))

done
echo "done."