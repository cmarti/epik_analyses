ngpu=1
out_dir="output"
activate="conda activate epik : module load cudnn8.1-cuda11.2/8.1.1.33 ;"

lr="0.05"
n=1000

gpu_options="--gpu -m $ngpu"
epik="python /grid/mccandlish/home_norepl/martigo/programs/epik/bin/EpiK.py -n $n -r $lr $gpu_options --train_noise --lr_decay"
submit="qsub -cwd -l gpu=$ngpu"
dataset='smn1'

for i in $(seq 1 5)
do
	trainx="datasets/$dataset.csv"
	cmd="$activate $epik $trainx"

	for kernel in Connectedness # Exponential Connectedness Jenga GeneralProduct 
	do	
		jid="$dataset.$i.$kernel"
		out=$out_dir/$dataset.full.$i.$kernel.test_pred.csv
		run="$cmd -k $kernel -o $out"
		echo "$run"  | $submit -N "$jid" -e "logs/$jid.err" -o "logs/$jid.out"
	done
done
