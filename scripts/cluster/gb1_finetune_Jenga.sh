ngpu=1
out_dir="output"
activate="conda activate epik : module load cudnn8.1-cuda11.2/8.1.1.33 ;"

lr="0.01"
# 0.00001 does not improve
n=500

gpu_options="--gpu -m $ngpu"
epik="python /grid/mccandlish/home_norepl/martigo/programs/epik/bin/EpiK.py -n $n -r $lr $gpu_options --train_noise --num_trace_samples 30 --cg_tol 0.5 --lr_decay --gamma 0.1"
submit="qsub -cwd -l gpu=$ngpu"
dataset='gb1'

for i in $(seq 1 3)
do
	trainx="datasets/$dataset.csv"
	#trainx="splits/$dataset.34.train.csv"
	cmd="$activate $epik $trainx"
	

	kernel="Jenga"
	jid="$dataset.$i.$kernel"
	out=$out_dir/$dataset.finetune.$lr.$i.$kernel.test_pred.csv
	params="$out_dir/$dataset.Connectedness.$i.$kernel.model_params.pth"
	run="$cmd -k $kernel -o $out --params $params"
	echo "$run"  | $submit -N "$jid" -e "logs/$jid.err" -o "logs/$jid.out"


done
