ngpu=1
out_dir="output"
activate="conda activate epik"
epik="EpiK -n 100"                           # n number of training iterations
submit="qsub -cwd -l gpu=$ngpu"              # command for job submission under SGE
gpu_options="--gpu -m $ngpu"                 # -s 1000 for partitioning

for dataset in $(grep -v '^#' datasets.txt)  # control which datasets are run through this file
do

for i in $(seq 0 4)
do
	trainx="splits/$dataset.$i.train.csv"
	testx="splits/$dataset.$i.test.txt"
	cmd="$activate; $epik $trainx -p $testx"

	for kernel in RBF Rho RhoPi # VC DP Rho RhoPi RBF ARD HetRBF HetARD
	do	
		jid="$kernel.$i.$dataset.gpu"
		run="$cmd -k $kernel -o $out_dir/$dataset.$i.$kernel.test_pred.csv $gpu_options"
		echo "$run" | $submit -N "$jid" -e "logs/$jid.err" -o "logs/$jid.out" # Comment before PIPE to only show the command

		# Uncomment next line to run in command line directly
		# $run  #
	done
done

done
