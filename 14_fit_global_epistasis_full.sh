out_dir="output"
activate="conda activate mavenn;"
mavenn="scripts/fit_mavenn.py -n 20000 -r 0.005 -l2 0"    # n number of training iterations and learning rate
submit="qsub -cwd -l mem_free=16G"                       # command for job submission under SGE

for dataset in $(grep -v '^#' datasets.txt)  # control which datasets are run through this file
do
	data="datasets/$dataset.csv"
	out="results/$dataset.global_epistasis"
	model="$out.model"
	cmd="$activate $mavenn -d $data -o $out -m $model"
	jid="ge.$dataset"
        echo "$cmd" | $submit -N "$jid" -e "logs/$jid.err" -o "logs/$jid.out"
done
