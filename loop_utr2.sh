#!/bin/bash
for f in $1/heimdall_32bit_*beam0.fil
do
    python /home/ofionnad/scripts/nenufar_utr_analysis.py $f
done