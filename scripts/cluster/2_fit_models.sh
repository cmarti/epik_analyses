ngpu=1
out_dir="output_new"
activate="conda activate epik ; module load cudnn8.1-cuda11.2/8.1.1.33 ;"
#activate=""

tc=10
lr=0.03
n=1000
t="1-60"
n_lanczos=600

lr=0.2
#lr=0.02
#n=500
#t="20-20"

gpu_options="--gpu -m $ngpu"     
epik="python /grid/mccandlish/home_norepl/martigo/programs/epik/bin/EpiK.py -n $n -r $lr $gpu_options --train_noise --n_lanczos $n_lanczos --num_trace_samples 100"
submit="qsub -cwd -l gpu=$ngpu -tc $tc -t $t" 

for dataset in qtls_li_hq
do


	for kernel in Additive Pairwise # Exponential # Connectedness # Jenga GeneralProduct
	do	

		trainx="splits/$dataset.\$SGE_TASK_ID.train.csv"
 		#testx="splits/$dataset.\$SGE_TASK_ID.test.txt"
                out="$out_dir/$dataset.\$SGE_TASK_ID.$kernel"

		cmd="$activate $epik $trainx -k $kernel -o $out"

		jid="$kernel.$dataset"
		sub="$submit -N $jid -e logs/$jid.err -o logs/$jid.out"
		echo "$cmd"  | $sub
	done
done
