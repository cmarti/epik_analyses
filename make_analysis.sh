source activate.sh

echo "=== Data preprocessing ==="
python scripts/process/smn1_calc_vj_covariances.py
python scripts/process/yeast_extract_qtls_prob.py
python scripts/process/yeast_extract_snps_annotations.py
python scripts/process/yeast_get_dataset.py
python scripts/process/yeast_get_pred.py

echo "=== Cross-validation analysis ==="
bash scripts/cluster/split_data.sh
bash scripts/cluster/fit_models_cv.sh
bash scripts/cluster/fit_global_epistasis_cv.sh
bash scripts/cluster/predict_models_cv.sh

echo "=== Full models ==="
bash scripts/cluster/fit_models.sh
bash scripts/cluster/fit_global_epistasis.sh
bash scripts/cluster/predict_models.sh
bash scripts/cluster/contrast_models.sh

echo "=== Processing results ==="
python scripts/process/calc_cv_curves.py
python scripts/process/calc_decay_factors.py
python scripts/process/gb1_calc_visualization.py
python scripts/process/gb1_calc_kernel_peaks.py
python scripts/process/aav_get_mut_effs.py
python scripts/process/yeast_process_results.py
python scripts/process/yeast_prep_visualization.py