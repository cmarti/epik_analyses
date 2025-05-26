ngpu=1
out_dir="output_new"
activate="conda activate epik : module load cudnn8.1-cuda11.2/8.1.1.33 ;"

lr="0.05"
n=500

gpu_options="--gpu -m $ngpu"
epik="python /grid/mccandlish/home_norepl/martigo/programs/epik/bin/EpiK.py -n $n -r $lr $gpu_options --num_trace_samples 50 --train_noise --n_lanczos 400"
submit="qsub -cwd -l gpu=$ngpu"
dataset='gb1'

for i in $(seq 1 2)
do
	trainx="datasets/$dataset.csv"
	cmd="$activate $epik $trainx"

	for kernel in Additive Pairwise VC # Exponential Connectedness Jenga GeneralProduct # Exponential Connectedness Jenga GeneralProduct 
	do	
		jid="$dataset.$i.$kernel"
		out=$out_dir/$dataset.full.$i.$kernel
		run="$cmd -k $kernel -o $out"
		echo "$run"  | $submit -N "$jid" -e "logs/$jid.err" -o "logs/$jid.out"
	done
done
