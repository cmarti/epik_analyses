# File and directory names
out_dir="output/models"
logs_dir="output/logs"
data_dir="data"

# Options
ngpu=1
tc=10
lr=0.01
n=500
t="1-60"
n_lanczos=600
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
		trainx="$data_dir/$dataset.csv"
		cmd="$activate $epik $trainx"

		for kernel in Exponential Connectedness Jenga GeneralProduct 
		do	
			out=$out_dir/$dataset.full.$i.$kernel.test_pred.csv

			jid="fit.$dataset.$i.$kernel"
			run="$cmd -k $kernel -o $out"
			echo "$run"  | $submit -N "$jid" -e "$logs_dir/$jid.err" -o "$logs_dir/$jid.out"
		done
	done

	dataset="qtls_li_hq"
	trainx="$data_dir/$dataset.csv"
        cmd="$activate $epik $trainx"

        for kernel in Exponential Connectedness
        do
                jid="$dataset.$i.$kernel"
                out=$out_dir/$dataset.full.$i.$kernel.test_pred.csv
                run="$cmd -k $kernel -o $out"
                echo "$run" | $submit -N "$jid" -e "$logs_dir/$jid.err" -o "$logs_dir/$jid.out"
        done
done
