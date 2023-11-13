out_dir="cpu"
activate="conda activate epik"
epik="EpiK -n 100"                           # n number of training iterations
submit="qsub -cwd -l mem_free=64G"            # command for job submission under SGE

for dataset in $(grep -v '^#' datasets.txt)  # control which datasets are run through this file
do

for i in $(seq 10 19)
do
	trainx="splits/$dataset.$i.train.csv"
	testx="splits/$dataset.$i.test.txt"
	cmd="$activate & $epik $trainx -p $testx"

	for kernel in Rho RhoPi # VC DP RBF ARD
	do	
		jid="$kernel.$i.$dataset"
		echo "$cmd -k $kernel -o $out_dir/$dataset.$i.$kernel.test_pred.csv" | $submit -N "$jid" -e "logs/$jid.cpu.err" -o "logs/$jid.cpu.out"
	done
done

done
