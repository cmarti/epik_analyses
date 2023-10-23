
dataset="gb1"
activate="conda activate gpmap"
submit="qsub -cwd -l mem_free=16G"

echo "$activate & evaluate_split_fits $dataset.csv -p $dataset -o $dataset.r2.csv" | $submit -N "r2"
