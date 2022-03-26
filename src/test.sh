#!/bin/bash

DES_BIN="../build-docker/desframework"

set -x

# GPU tests for compute batch sizes

#for cbs in 50 500 1000 2000 5000 10000 20000 50000; do

#        echo ------------------------------ compute batch size = $cbs ------------------------------ | tee -a results_exp.txt

#        kill `pidof parallelFramework`
#        mpirun -n 2 --bind-to none --mca btl_base_warn_component_unused 0 ${DES_BIN} -gs 3 -gpu -cbs $cbs -ssl 0 | tee -a results_exp.txt

#done

#mv results_exp.txt results_gpu.txt

# CPU tests for cpu compute batch sizes

#for ccbs in 100000 1000000; do
#
#        echo ------------------ ccbs = $ccbs ------------------- | tee -a results_exp.txt
#
#        kill `pidof parallelFramework`
#        mpirun -n 2 --bind-to none --mca btl_base_warn_component_unused 0 ${DES_BIN} -gs 3 -cpu -ccbs $ccbs -ssl 0 | tee -a results_exp.txt
#
#        if [[ $? != 0 ]]; then
#        	exit 1
#        fi
#
#done
#
#mv results_exp.txt results_cpu2.txt

# CPU+GPU tests for slave batch sizes = 1e5 -> 1e11 and both balancing types (last, average)

ccbs=10000
cbs=20

for tba in 0 1; do
for sbs in 100000 1000000 10000000 100000000 1000000000 10000000000 100000000000; do

        echo ------------------ tba = $tba, sbs = $sbs ------------------- | tee -a results_exp.txt

        kill `pidof parallelFramework`
        mpirun -n 2 --bind-to none --mca btl_base_warn_component_unused 0 ${DES_BIN} -gs 3 -both -tba $tba -sbs $sbs -cbs $cbs -ccbs $ccbs -ssl 0 | tee -a results_exp.txt

done
done

mv results_exp.txt results_cpugpu.txt

# CPU+GPU tests for slave batch size factor = 1/4, 1/8, 1/16, 1/32, 1/64 and both balancing types (last, average)

#ccbs=10000
#cbs=20
#for tba in 1; do
#for sbsf in 0.25 0.125 0.0625 0.03125 0.015625; do

#        echo ------------------ tba = $tba, sbsf = $sbsf ------------------- | tee -a results_exp.txt

#        kill `pidof parallelFramework`
#        mpirun -n 2 --bind-to none --mca btl_base_warn_component_unused 0 ${DES_BIN} -gs 3 -both -tba $tba -sbsf $sbsf -cbs $cbs -ccbs $ccbs -ssl 0 | tee -a results_exp.txt
        
#        if [[ $? != 0 ]]; then
#        	exit 1
#        fi

#done
#done

#mv results_exp.txt results_both.txt


