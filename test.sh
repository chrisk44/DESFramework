#!/bin/bash

for sbs in 10000 100000 1000000 10000000 100000000 1000000000; do

	for cbs in 20 50 100 200 500 1000 2000 5000 10000 20000 50000; do

		echo ------------------------------ cbs = $cbs ------------------------------ | tee -a results_exp.txt

		for m in 1 2 3 4; do

			for g in 3 4 5 6; do

				kill `pidof parallelFramework`
				mpirun -n 2 --bind-to none --mca btl_base_warn_component_unused 0 ./parallelFramework -gs $g -ge $g -ms $m -me $m -gpu -cbs $cbs | tee -a results_exp.txt

			done

		done

	done

done
