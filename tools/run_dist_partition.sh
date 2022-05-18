#!/bin/bash

generatePartition(){
	add=1
	val=`expr $1 + $add`
	# ViT Base [1,48]
	echo "1,$1,$val,48"
	# ViT Large [1,96]
	# echo "1,$1,$val,96"
}

for bit in 2 4 6 8
do
	# ViT Base [1,47]; ViT Large [1,95]
	for pt in `seq 1 47`
	do
		pt=$(generatePartition $pt)
		# python get_distribution.py -pt $pt -q $bit,$bit --baseline
		sbatch upload_run_dist.job $pt $bit
	done
done		
