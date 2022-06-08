#!/bin/bash

fpath=$1
beam_num=$2
mkdir -p $1/beam_${beam_num}

for file in $fpath/heimdall_32bit_*beam${beam_num}.fil; do
    digifil -b 8 -t 24 -f 32 $file -o ${file/32bit_fullres/8bit_downres}
    # heimdall -f ${file/32bit_fullres/8bit_downres} -output_dir $1/beam_${beam_num} -rfi_no_narrow -rfi_no_broad -dm 0 0 -dm_tol 1.0001
done

# cd $1/beam_${beam_num}
# cat *.cand > beam_${beam_num}.cand
# cd -

exit