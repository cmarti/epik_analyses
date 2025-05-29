ngpu=1
out_dir="output_new"
activate="conda activate epik ; module load cudnn8.1-cuda11.2/8.1.1.33 ;"
#activate=""
epik="python /grid/mccandlish/home_norepl/martigo/programs/epik/bin/EpiK.py -n 0 --train_noise"                             # n number of training iterations and learning rate
submit="qsub -cwd -l gpu=$ngpu"              # command for job submission under SGE
gpu_options="--gpu -m $ngpu"                 # -s 1000 for partitioning


#seq0="CAGUAAGU"
#seq0="DEEEIRTTNPVATEQYGSVSTNLQRGNR"
#seq0="VDGV"

kernel="Jenga"
dataset="aav"

for seq0 in DEEEIRTTNPVATEQYGSVSTNLQRGNR DEEEIRTTQPVATEQYGSVSTNLQRGNR DEEEIRTTNPVATEQFGSVSTNLQRGNR DEEEIRTTNPVATEQCGSVSTNLQRGNR DEEEIRTTNPVATEQYGSVSTNLQRGER DEEEIRTTNPVATEQYGSVSTNLQEGER
do

	trainx="datasets/$dataset.csv"
	cmd="$activate $epik $trainx -s $seq0"

	jid="contrast.$dataset"
	sub="$submit -N $jid -e logs/$jid.err -o logs/$jid.out"
	params="$out_dir/$dataset.full.1.$kernel.model_params.pth"

	out="$out_dir/$dataset.$kernel"
	run="$cmd -k $kernel -o $out $gpu_options --params $params --max_contrasts 10 --calc_variance" # --calc_epi_coef"
	echo "$run" | $sub

done
