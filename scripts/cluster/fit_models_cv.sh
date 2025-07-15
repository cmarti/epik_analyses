# File and directory names
out_dir="output/models"
logs_dir="output/logs"
splits_dir="data/splits"

# Options
ngpu=1
tc=10
lr=0.01
n=1000
t="1-60"
n_lanczos=600
gpu_options="--gpu -m $ngpu"     

# Commands
activate="conda activate epik ; module load cudnn8.1-cuda11.2/8.1.1.33 ;"
epik="python /grid/mccandlish/home/martigo/programs/epik/epik/bin/EpiK.py -n $n -r $lr $gpu_options --train_noise --n_lanczos $n_lanczos --num_trace_samples 100"
submit="qsub -cwd -l gpu=$ngpu -tc $tc -t $t" 

# Run
for dataset in gb1 smn1
do

	for kernel in Additive Pairwise VC Exponential Connectedness Jenga GeneralProduct
	do	

		trainx="$splits_dir/$dataset.\$SGE_TASK_ID.train.csv"
        out="$out_dir/$dataset.\$SGE_TASK_ID.$kernel"
		
        cmd="$activate $epik $trainx -k $kernel -o $out"
		jid="fit.$kernel.$dataset"
		echo "$cmd" | $submit -N $jid -e $logs_dir/$jid.err -o $logs_dir/$jid.out
	done
done

dataset="aav"
for kernel in Additive Pairwise Exponential Connectedness Jenga GeneralProduct
do	

    trainx="$splits_dir/$dataset.\$SGE_TASK_ID.train.csv"
    out="$out_dir/$dataset.\$SGE_TASK_ID.$kernel"
    
    cmd="$activate $epik $trainx -k $kernel -o $out"
    jid="fit.$kernel.$dataset"
    echo "$cmd" | $submit -N $jid -e $logs_dir/$jid.err -o $logs_dir/$jid.out
done

dataset="qtls_li_hq"
for kernel in Additive Pairwise Exponential Connectedness
do	

    trainx="$splits_dir/$dataset.\$SGE_TASK_ID.train.csv"
    out="$out_dir/$dataset.\$SGE_TASK_ID.$kernel"
    
    cmd="$activate $epik $trainx -k $kernel -o $out"
    jid="fit.$kernel.$dataset"
    echo "$cmd" | $submit -N $jid -e $logs_dir/$jid.err -o $logs_dir/$jid.out
done
