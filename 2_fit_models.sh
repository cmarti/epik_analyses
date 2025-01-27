ngpu=1
out_dir="output"
activate="conda activate epik ; module load cudnn8.1-cuda11.2/8.1.1.33 ;"
#activate=""

tc=10
lr=0.02
n=1000
t="1-60"

#lr=0.02
#n=10
#t="50-50"

gpu_options="--gpu -m $ngpu"     
epik="python /grid/mccandlish/home_norepl/martigo/programs/epik/bin/EpiK.py -n $n -r $lr $gpu_options --train_noise"
submit="qsub -cwd -l gpu=$ngpu -tc $tc -t $t" 

for dataset in smn1 # gb1 aav qtls_li_hq
do


	for kernel in Additive Pairwise # Exponential Connectedness Jenga GeneralProduct VC
	do	

		trainx="splits/$dataset.\$SGE_TASK_ID.train.csv"
 		#testx="splits/$dataset.\$SGE_TASK_ID.test.txt"
                out="$out_dir/$dataset.\$SGE_TASK_ID.$kernel.test_pred.csv"
		#params="$out.model_params.pth"

 		#cmd="$activate $epik $trainx -p $testx -k $kernel -o $out --params $params"
		cmd="$activate $epik $trainx -k $kernel -o $out"

		jid="$kernel.$dataset"
		sub="$submit -N $jid -e logs/$jid.err -o logs/$jid.out"
		echo "$cmd"  | $sub
	done
done
