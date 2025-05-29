out_dir="output"
activate="conda activate epik ; module load cudnn8.1-cuda11.2/8.1.1.33 ; export PYTHONPATH=$PYTHONPATH:/grid/mccandlish/home_norepl/martigo/projects/epik_analysis ;"

cmd="python scripts/calc_mlls_test_smn1.py"
submit="qsub -cwd -l gpu=1" 

jid="mlls_smn1"
sub="$submit -N $jid -e logs/$jid.err -o logs/$jid.out"
echo "$cmd"  | $sub
