ngpu=1
out_dir="output"
activate="conda activate mavenn ; "

lr=0.01
l2=0
n=5000
t="1-60"

#t="13-13"

mavenn="scripts/fit_mavenn.py -n $n -r $lr -l2 $l2"    # n number of training iterations and learning rate
submit="qsub -cwd -l mem_free=16G -tc 20 -t $t"          # command for job submission under SGE

for dataset in smn1 # aav gb1
do

	trainx="splits/$dataset.\$SGE_TASK_ID.train.csv"
	testx="splits/$dataset.\$SGE_TASK_ID.test.txt"
        out="$out_dir/$dataset.\$SGE_TASK_ID.global_epistasis"
	model="$out.model"

	cmd="$activate $mavenn -d $trainx -o $out -m $model -p $testx"
	jid="ge.$dataset"
	sub="$submit -N $jid -e logs/$jid.err -o logs/$jid.out"
	echo "$cmd"  | $sub

done
