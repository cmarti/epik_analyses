# File and directory names
out_dir="output/models"
logs_dir="output/logs"
splits_dir="data/splits"

# Options
lr=0.01
l2=0
n=5000
t="1-60"

# Commands
activate="conda activate mavenn ; "
mavenn="scripts/fit_mavenn.py -n $n -r $lr -l2 $l2"
submit="qsub -cwd -l mem_free=16G -tc 20 -t $t"

# Run
for dataset in gb1 smn1 aav
do

	trainx="$splits_dir/$dataset.\$SGE_TASK_ID.train.csv"
	testx="$splits_dir/$dataset.\$SGE_TASK_ID.test.txt"
    out="$out_dir/$dataset.\$SGE_TASK_ID.global_epistasis"
	model="$out.model"

	cmd="$activate $mavenn -d $trainx -o $out -m $model -p $testx"
	jid="ge.$dataset"
	echo "$cmd" | $submit -N $jid -e logs/$jid.err -o logs/$jid.out
done
