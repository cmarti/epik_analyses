ngpu=1
out_dir="output_new"
activate="conda activate epik : module load cudnn8.1-cuda11.2/8.1.1.33 ;"

lr="0.02"
n="0"

gpu_options="--gpu -m $ngpu"
epik="python /grid/mccandlish/home_norepl/martigo/programs/epik/bin/EpiK.py -n $n -r $lr $gpu_options --train_noise"
submit="qsub -cwd -l gpu=$ngpu"

dataset='gb1'
kernel="Jenga"

label=$out_dir/$dataset.full.1.$kernel
params="$label.model_params.pth"
out="$label.full_pred.csv"

train="datasets/$dataset.csv"
pred="datasets/gb1.seqs.txt"

jid="pred.$dataset.$kernel"
run="$activate $epik $train --params $params -o $out -p $pred -k $kernel"
echo "$run"  | $submit -N "$jid" -e "logs/$jid.err" -o "logs/$jid.out"

exit

for i in $(seq 1 5)
do
	for dataset in smn1 aav gb1
	do
		trainx="datasets/$dataset.csv"
		cmd="$activate $epik $trainx"

		for kernel in Exponential Connectedness Jenga GeneralProduct 
		do	
			jid="p$dataset.$i.$kernel"
			out=$out_dir/$dataset.full.$i.$kernel.test_pred.csv
			params="$out.max_evid.model_params.pth"

			run="$cmd -k $kernel -o $out --params $params"

			echo "$run"  | $submit -N "$jid" -e "logs/$jid.err" -o "logs/$jid.out"
		done
	done

	dataset="qtls_li_hq"
	trainx="datasets/$dataset.csv"
        cmd="$activate $epik $trainx"

        for kernel in Exponential Connectedness
        do
                jid="p$dataset.$i.$kernel"
                out=$out_dir/$dataset.full.$i.$kernel.test_pred.csv
                params="$out.max_evid.model_params.pth"

                run="$cmd -k $kernel -o $out --params $params"

                echo "$run" | $submit -N "$jid" -e "logs/$jid.err" -o "logs/$jid.out"
        done

done
