# File and directory names
out_dir="output/models"
logs_dir="output/logs"
data_dir="data"

# Options
lr=0.01
l2=0
n=1000

# Commands
activate="conda activate mavenn ; "
mavenn="scripts/fit_mavenn.py -n $n -r $lr -l2 $l2"
submit="qsub -cwd -l mem_free=16G"

# Runs
dataset=smn1
trainx="$data/$dataset.csv"
out="$out_dir/$dataset.global_epistasis"
model="$out.model"

jid="ge.$dataset"
cmd="$activate $mavenn -d $trainx -o $out -m $model"
echo "$cmd"  | $submit -N $jid -e $logs_dir/$jid.err -o $logs_dir/$jid.out
