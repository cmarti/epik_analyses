# File and directory names
out_dir="output/models"
logs_dir="output/logs"
data_dir="data"

# Options
ngpu=1
tc=10
gpu_options="--gpu -m $ngpu"    

# Commands
activate="conda activate epik ; module load cudnn8.1-cuda11.2/8.1.1.33 ;"
epik="python /grid/mccandlish/home/martigo/programs/epik/epik/bin/EpiK.py -n $n -r $lr $gpu_options --train_noise"
submit="qsub -cwd -l gpu=$ngpu"

# Run
for i in $(seq 1 5)
do
	for dataset in gb1 smn1 aav
	do
		kernel="Jenga"
		trainx="$data_dir/$dataset.csv"
        testx="$data_dir/$dataset.seqs.txt"
        out=$out_dir/$dataset.full.$i.$kernel.test_pred.csv
        params="$out.model_params.pth"
		
        jid="pred.$dataset.$i.$kernel"
        cmd="$activate $epik $trainx -p $testx -k $kernel -o $out --params $params"
        echo "$cmd"  | $submit -N "$jid" -e "$logs_dir/$jid.err" -o "$logs_dir/$jid.out"
	done

	dataset="qtls_li_hq"
    kernel="Connectedness"

	trainx="$data_dir/$dataset.csv"
    testx="$data_dir/$dataset.seqs.txt"
    out=$out_dir/$dataset.full.$i.$kernel.test_pred.csv
    params="$out.model_params.pth"
    
    jid="pred.$dataset.$i.$kernel"
    cmd="$activate $epik $trainx -p $testx -k $kernel -o $out --params $params"
    echo "$cmd"  | $submit -N "$jid" -e "$logs_dir/$jid.err" -o "$logs_dir/$jid.out"

done
