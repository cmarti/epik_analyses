out_dir="output_new"
activate="conda activate mavenn ; "

lr=0.01
l2=0
n=1000

mavenn="scripts/fit_mavenn.py -n $n -r $lr -l2 $l2"    # n number of training iterations and learning rate
submit="qsub -cwd -l mem_free=16G"          # command for job submission under SGE

for dataset in smn1 # aav gb1
do

	trainx="datasets/$dataset.csv"
        out="$out_dir/$dataset.global_epistasis"
	model="$out.model"

	cmd="$activate $mavenn -d $trainx -o $out -m $model"
	jid="ge.$dataset"
	sub="$submit -N $jid -e logs/$jid.err -o logs/$jid.out"
	echo "$cmd"  | $sub

done
