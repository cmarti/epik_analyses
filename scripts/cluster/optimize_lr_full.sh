ngpu=1
out_dir="output"
activate="conda activate epik : module load cudnn8.1-cuda11.2/8.1.1.33 ;"

n="500"

gpu_options="--gpu -m $ngpu"
epik="python /grid/mccandlish/home_norepl/martigo/programs/epik/bin/EpiK.py -n $n  $gpu_options --train_noise --num_trace_samples 200"
submit="qsub -cwd -l gpu=$ngpu"

dataset="gb1"

trainx="datasets/$dataset.csv"
trainx="splits/$dataset.34.train.csv"
cmd="$activate $epik $trainx"
kernel="Jenga"

lrs="0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001"
#lrs="0.001 0.0005 0.0001 0.00005 0.000001 0.000005"
optimizer="Adam"

for lr in $lrs
do
	jid="lr"
	out="$out_dir/$dataset.finetune.$optimizer.opt_lr.$lr"
	params="$out_dir/$dataset.Connectedness1.$kernel.model_params.pth"
	run="$cmd -k $kernel -o $out -r $lr --params $params"
	echo "$run"  | $submit -N "$jid" -e "logs/$jid.err" -o "logs/$jid.out"
done
