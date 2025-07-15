# File and directory names
out_dir="output/models"
logs_dir="output/logs"
data_dir="data"

# Options
ngpu=1
gpu_options="--gpu -m $ngpu"    

# Commands
activate="conda activate epik ; module load cudnn8.1-cuda11.2/8.1.1.33 ;"
epik="python /grid/mccandlish/home/martigo/programs/epik/epik/bin/EpiK.py -n 0 --train_noise $gpu_options --max_contrasts 10 --calc_variance --n_lanczos 100 --num_trace_samples 500"
submit="qsub -cwd -l gpu=$ngpu"


# Run AAV contrasts
dataset="aav"
kernel="Jenga"

trainx="$data_dir/$dataset.csv"
params="$out_dir/$dataset.full.1.$kernel.model_params.pth"
out="$out_dir/$dataset.$kernel"

for seq0 in DEEEIRTTNPVATEQYGSVSTNLQRGNR DEEEIRTTQPVATEQYGSVSTNLQRGNR DEEEIRTTNPVATEQFGSVSTNLQRGNR DEEEIRTTNPVATEQCGSVSTNLQRGNR DEEEIRTTNPVATEQYGSVSTNLQRGER DEEEIRTTNPVATEQYGSVSTNLQEGER
do
	jid="contrast.$dataset"
    cmd="$activate $epik $trainx -k $kernel --params $params -o $out -s $seq0"
	echo "$cmd" | $submit -N $jid -e $logs_dir/$jid.err -o $logs_dir/$jid.out

done

# Run yeast contrasts
dataset="qtls_li_hq"
kernel="Connectedness"

trainx="$data_dir/$dataset.csv"
params="$out_dir/$dataset.full.1.$kernel.model_params.pth"
out="$out_dir/$dataset.$kernel"

for seq0 in AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA AAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA BBBBBBBBBBBBABBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
do
	jid="contrast.$dataset"
    cmd="$activate $epik $trainx -k $kernel --params $params -o $out -s $seq0"
	echo "$cmd" | $submit -N $jid -e $logs_dir/$jid.err -o $logs_dir/$jid.out

done
