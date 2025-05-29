for dataset in gb1 smn1 aav qtls_li_hq
do
	split_data "data/$dataset.csv" --ps data/training_p.txt -r 3 -o splits.csv -p data/splits/$dataset --seed 0 -m 10000 -f csv
done

