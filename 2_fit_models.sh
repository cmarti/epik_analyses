ngpu=1
out_dir="output"
activate="conda activate epik"
epik="EpiK -n 100"                           # n number of training iterations
submit="qsub -cwd -l gpu=$ngpu"              # command for job submission under SGE
gpu_options="--gpu -m $ngpu"                 # -s 1000 for partitioning

for dataset in $(grep -v '^#' datasets.txt)  # control which datasets are run through this file
do

for i in $(seq 2 5)
do
	trainx="splits/$dataset.$i.train.csv"
	testx="splits/$dataset.$i.test.txt"
	cmd="$activate; $epik $trainx -p $testx"

	for kernel in Rho RhoPi # VC DP Rho RhoPi RBF ARD
	do	
		jid="$kernel.$i.$dataset"
		echo "$cmd -k $kernel -o $out_dir/$dataset.$i.$kernel.test_pred.csv $gpu_options" | $submit -N "$jid" -e "logs/$jid.err" -o "logs/$jid.out"
	done
done

done
