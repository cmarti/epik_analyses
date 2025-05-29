ngpu=1
out_dir="output_new"
activate="conda activate epik ; module load cudnn8.1-cuda11.2/8.1.1.33 ;"

epik="python /grid/mccandlish/home_norepl/martigo/programs/epik/bin/EpiK.py -n 0 --train_noise"
submit="qsub -cwd -l gpu=$ngpu"
gpu_options="--gpu -m $ngpu"

dataset="qtls_li_hq"
kernel="Connectedness"
trainx="datasets/$dataset.csv"
params="$out_dir/$dataset.full.1.$kernel.model_params.pth"

out="$out_dir/$dataset.$kernel.2.pred.csv"
pred="datasets/$dataset.seqs.txt"
cmd="$activate $epik $trainx -k $kernel --params $params --max_contrasts 10 --calc_variance  $gpu_options -o $out --n_lanczos 100 --num_trace_samples 500"

jid="pred.$dataset"
sub="$submit -N $jid -e logs/$jid.err -o logs/$jid.out"
run="$cmd -p $pred"
echo $run | $sub
