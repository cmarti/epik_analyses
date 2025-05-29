ngpu=1
out_dir="output"
activate="conda activate epik : module load cudnn8.1-cuda11.2/8.1.1.33 ;"

n="1000"

gpu_options="--gpu -m $ngpu"
epik="python /grid/mccandlish/home_norepl/martigo/programs/epik/bin/EpiK.py -n $n  $gpu_options" # --train_noise"
submit="qsub -cwd -l gpu=$ngpu"

dataset="gb1"
i="16"

trainx="splits/$dataset.$i.train.csv"
cmd="$activate $epik $trainx"
kernel="Jenga"

lrs="0.1 0.05 0.01 0.005 0.001 0.0005"
#lrs="0.001 0.0005 0.0001 0.00005 0.000001 0.000005"
optimizer="Adam"

for lr in $lrs
do
	for i in $(seq 1 3) 
	do	
		jid="lr"
		out="$out_dir/$dataset.$optimizer.opt_lr.$lr.$i"
		run="$cmd -k $kernel -o $out -r $lr"
		echo "$run"  | $submit -N "$jid" -e "logs/$jid.err" -o "logs/$jid.out"
	done
done
