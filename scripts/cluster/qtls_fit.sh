ngpu=1
out_dir="output_new"
activate="conda activate epik : module load cudnn8.1-cuda11.2/8.1.1.33 ;"

lr="0.03"
n=1000

#n=100

gpu_options="--gpu -m $ngpu"
epik="python /grid/mccandlish/home_norepl/martigo/programs/epik/bin/EpiK.py -n $n -r $lr $gpu_options --train_noise --n_lanczos 100 --num_trace_samples 500"
submit="qsub -cwd -l gpu=$ngpu"
dataset='qtls_li_hq'

for i in $(seq 1 2)
do
	trainx="datasets/$dataset.csv"
	cmd="$activate $epik $trainx"

	for kernel in Exponential # Connectedness 
	do	
		jid="$dataset.$i.$kernel"
		out=$out_dir/$dataset.full.$i.$kernel
		params=$out.model_params.pth
		run="$cmd -k $kernel -o $out" # --params $params"
		echo "$run"  | $submit -N "$jid" -e "logs/$jid.err" -o "logs/$jid.out"
	done
done
