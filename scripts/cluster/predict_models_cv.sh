# File and directory names
out_dir="output/models"
logs_dir="output/logs"
splits_dir="data/splits"

# Options
tc=10
t="1-60"

# Commands
activate="conda activate epik ; module load cudnn8.1-cuda11.2/8.1.1.33 ;"
gpu_options="--gpu -m $ngpu"     
epik="python /grid/mccandlish/home_norepl/martigo/programs/epik/bin/EpiK.py $gpu_options --train_noise -n 0"
submit="qsub -cwd -l gpu=$ngpu -tc $tc -t $t" 

# Run
for dataset in gb1 smn1
do
	for kernel in Additive Pairwise VC Exponential Connectedness Jenga GeneralProduct
	do	

		trainx="$splits_dir/$dataset.\$SGE_TASK_ID.train.csv"
 		testx="$splits_dir/$dataset.\$SGE_TASK_ID.test.txt"
        out="$out_dir/$dataset.\$SGE_TASK_ID.$kernel"
		params="$out.model_params.pth"

		cmd="$activate $epik $trainx -k $kernel -p $testx -o $out.test_pred.csv --params $params"
		jid="pred.$dataset.$kernel"
		echo "$cmd"  | $submit -N $jid -e $logs_dir/$jid.err -o $logs_dir/$jid.out
	done
done


dataset="aav"
for kernel in Additive Pairwise Exponential Connectedness Jenga GeneralProduct
do	

    trainx="$splits_dir/$dataset.\$SGE_TASK_ID.train.csv"
    testx="$splits_dir/$dataset.\$SGE_TASK_ID.test.txt"
    out="$out_dir/$dataset.\$SGE_TASK_ID.$kernel"
    params="$out.model_params.pth"

    cmd="$activate $epik $trainx -k $kernel -p $testx -o $out.test_pred.csv --params $params"
    jid="pred.$dataset.$kernel"
    echo "$cmd"  | $submit -N $jid -e $logs_dir/$jid.err -o $logs_dir/$jid.out
done

dataset="qtls_li_hq"
for kernel in Additive Pairwise Exponential Connectedness
do	

    trainx="$splits_dir/$dataset.\$SGE_TASK_ID.train.csv"
    testx="$splits_dir/$dataset.\$SGE_TASK_ID.test.txt"
    out="$out_dir/$dataset.\$SGE_TASK_ID.$kernel"
    params="$out.model_params.pth"

    cmd="$activate $epik $trainx -k $kernel -p $testx -o $out.test_pred.csv --params $params"
    jid="pred.$dataset.$kernel"
    echo "$cmd"  | $submit -N $jid -e $logs_dir/$jid.err -o $logs_dir/$jid.out
done