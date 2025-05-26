ngpu=1
out_dir="output"
activate="conda activate epik : module load cudnn8.1-cuda11.2/8.1.1.33 ;"

n="1000"
lr="0.01"

gpu_options="--gpu -m $ngpu"
epik="python /grid/mccandlish/home_norepl/martigo/programs/epik/bin/EpiK.py -n $n  $gpu_options" # --train_noise"
submit="qsub -cwd -l gpu=$ngpu"

dataset="smn1"
i="34"

#dataset="gb1"
#i="16"

#dataset="aav"
#i="24"

trainx="splits/$dataset.$i.train.csv"
cmd="$activate $epik $trainx"
kernel="Jenga"

gammas="0.01 0.1 0.2 0.5 0.8 0.9"
optimizer="Adam"

for gamma in $gammas
do
	for i in $(seq 1 3) 
	do	
		jid="gamma"
		out="$out_dir/$dataset.$optimizer.opt_gamma.$gamma.$i"
		run="$cmd -k $kernel -o $out -r $lr --gamma $gamma --lr_decay"
		echo "$run"  | $submit -N "$jid" -e "logs/$jid.err" -o "logs/$jid.out"
	done
done
