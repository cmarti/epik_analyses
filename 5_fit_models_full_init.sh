ngpu=1
out_dir="output"
activate="conda activate epik : module load cudnn8.1-cuda11.2/8.1.1.33 ;"
activate=""

lr="0.02"
n="500"
ptn="200"

gpu_options="--gpu -m $ngpu"
epik="python /grid/mccandlish/home_norepl/martigo/programs/epik/bin/EpiK.py -n $n -r $lr $gpu_options --train_noise" # --pre_train_n_iter $ptn"
submit="qsub -cwd -l gpu=$ngpu"



for dataset in gb1 aav
do
	for i in $(seq 1 5)
	do
		trainx="datasets/$dataset.csv"
		trainx="splits/$dataset.24.train.csv"
		#testx="datasets/$dataset.seqs.txt"

		cmd="$activate $epik $trainx" # -p $testx"

		for kernel1 in Exponential Connectedness GeneralProduct 
		do	
			for kernel2 in Exponential Connectedness Jenga GeneralProduct 
			do
				jid="$dataset.$i.$kernel1.$kernel2"
				out="$out_dir/$dataset.$kernel1.$i.$kernel2.test_pred.csv"
				params="$out_dir/$dataset.$kernel1$i.$kernel2.model_params.pth"
				run="$cmd -k $kernel2 -o $out --params $params"
				echo "$run" | $submit -N "$jid" -e "logs/$jid.err" -o "logs/$jid.out"
				#exit
			done
		done
	done
done
