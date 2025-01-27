ngpu=1
out_dir="output"
activate="conda activate epik : module load cudnn8.1-cuda11.2/8.1.1.33 ;"

lr="0.01"
n=1000

gpu_options="--gpu -m $ngpu"
epik="python /grid/mccandlish/home_norepl/martigo/programs/epik/bin/EpiK.py -n $n -r $lr $gpu_options" # --train_noise"
submit="qsub -cwd -l gpu=$ngpu"


for i in $(seq 51 55)
do
	dataset="gb1"
	trainx="splits/$dataset.24.train.csv"
	cmd="$activate $epik $trainx"

	for kernel in Jenga # GeneralProduct 
	do	
		jid="test.$kernel"
		out="$out_dir/$dataset.opt_test_noise.$i.$kernel.test_pred.csv"
		params="$out_dir/$dataset.opt_test_noise.$i.$kernel.test_pred.csv.max_evid.model_params.pth"
		run="$cmd -k $kernel -o $out" # --params $params"
		echo "$run"  | $submit -N "$jid" -e "logs/$jid.err" -o "logs/$jid.out"
	done
done
