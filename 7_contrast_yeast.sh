ngpu=1
out_dir="output"
activate="conda activate epik ; module load cudnn8.1-cuda11.2/8.1.1.33 ;"

epik="python /grid/mccandlish/home_norepl/martigo/programs/epik/bin/EpiK.py -n 0"
submit="qsub -cwd -l gpu=$ngpu"
gpu_options="--gpu -m $ngpu"

dataset="qtls_li_hq"
kernel="Connectedness"
trainx="datasets/$dataset.csv"
cmd="$activate $epik $trainx -s $seq0"

for seq0 in "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,AAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,BBBBBBBBBBBBABBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB,BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
do
	jid="contrast.$dataset"
	sub="$submit -N $jid -e logs/$jid.err -o logs/$jid.out"
	params="$out_dir/$dataset.$kernel.test_pred.csv.model_params.pth"

	out="$out_dir/$dataset.$kernel"
	run="$cmd -k $kernel -o $out $gpu_options --params $params --max_contrasts 10 --calc_variance --calc_epi_coef"
	echo "$run" | $sub

done
