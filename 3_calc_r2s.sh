
dataset="gb1"
activate="conda activate gpmap"
submit="qsub -cwd -l mem_free=16G"

for dataset in $(grep -v '^#' datasets.txt)  # control which datasets are run through this file
do
	echo "$activate & evaluate_split_fits datasets/$dataset.csv -p output/$dataset -o $dataset.r2.csv" # | $submit -N "r2.$dataset" -e "logs/r2.$dataset.err" -o "logs/r2.$dataset.out"

done
