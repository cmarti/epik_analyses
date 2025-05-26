for dataset in aav # qtls_li_hq # $(grep -v '^#' datasets.txt)
do
	split_data "datasets/$dataset.csv" --ps training_p.txt -r 3 -o splits.csv -p splits/$dataset --seed 0 -m 10000 -f csv
done

