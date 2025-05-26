ngpu=1
out_dir="output_new"
activate="conda activate epik ; module load cudnn8.1-cuda11.2/8.1.1.33 ;"
#activate=""

tc=10
t="1-60"

#lr=0.02
#n=10
t="56-56"

gpu_options="--gpu -m $ngpu"     
epik="python /grid/mccandlish/home_norepl/martigo/programs/epik/bin/EpiK.py $gpu_options --train_noise -n 0"
submit="qsub -cwd -l gpu=$ngpu -tc $tc -t $t" 

for dataset in aav # smn1 gb1
do


	for kernel in GeneralProduct # Additive Pairwise Exponential Connectedness # Jenga GeneralProduct # VC
	do	

		trainx="splits/$dataset.\$SGE_TASK_ID.train.csv"
 		testx="splits/$dataset.\$SGE_TASK_ID.test.txt"
                out="$out_dir/$dataset.\$SGE_TASK_ID.$kernel"
		params="$out.model_params.pth"

		cmd="$activate $epik $trainx -k $kernel -p $testx -o $out.test_pred.csv --params $params"

		jid="pred.$dataset.$kernel"
		sub="$submit -N $jid -e logs/$jid.err -o logs/$jid.out"
		echo "$cmd"  | $sub
	done
done
