#!/bin/bash

for sbs in 10000 100000 1000000 10000000 100000000 1000000000; do

	for cbs in 20 50 100 200 500 1000 2000 5000 10000 20000 50000; do

		kill `pidof parallelFramework`
		echo ------------------------------ cbs = $cbs ------------------------------ | tee -a results_exp.txt
		mpirun -n 2 --bind-to none --mca btl_base_warn_component_unused 0 ./parallelFramework -gs 3 -me 3 -gpu -cbs $cbs | tee -a results_exp.txt
		mpirun -n 2 --bind-to none --mca btl_base_warn_component_unused 0 ./parallelFramework -gs 2 -ms 4 -gpu -cbs $cbs | tee -a results_exp.txt

	done

done
