for dataset in $(grep -v '^#' datasets.txt)
do
	split_data "datasets/$dataset.csv" -r 5 -o splits.csv -p splits/$dataset --seed 0 -m 10000 -f csv
done
