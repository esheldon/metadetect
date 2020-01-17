#!/bin/bash

export OMP_NUM_THREADS=1

nepoch=$1
ntrial=$2
seed=$3
output=$4
logfile=${output}.log

tmpdir=$_CONDOR_SCRATCH_DIR
tmplog=${tmpdir}/logfile

python /astro/u/esheldon/lensing/test-lsst-mdet/lsst_sim.py \
    --nepochs ${nepoch} \
    --noise 1 \
    --ntrial ${ntrial} \
    --seed ${seed} \
    --output ${output} &> ${tmplog}

mv -fv $tmplog $logfile
