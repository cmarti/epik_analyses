source activate.sh

echo "=== Figure 1: GB1 dataset ==="
python scripts/figures/gb1/plot_cv_curve.py
python scripts/figures/gb1/plot_decay_rates_jenga.py
python scripts/figures/gb1/plot_decay_rates_gp.py
python scripts/figures/gb1/plot_jenga_visualization.py

echo "=== Supplementary figures: GB1 dataset ==="
python scripts/figures/gb1/supp_plot_decay_rates.py
python scripts/figures/gb1/supp_plot_kernels_visualization.py

echo "=== Figure 2: SMN1 dataset ==="
python scripts/figures/smn1/plot_cv_curves_dist_corr.py
python scripts/figures/smn1/plot_decay_rates.py
python scripts/figures/smn1/plot_predictions.py

echo "=== Supplementary figures: SMN1 dataset ==="
python scripts/figures/smn1/supp_plot_smn1_ge_params.py
python scripts/figures/smn1/supp_plot_decay_rates.py

echo "=== Figure 3: AAV2 dataset ==="
python scripts/figures/aav/plot_cv_curve.py
python scripts/figures/aav/plot_decay_rates.py
python scripts/figures/aav/plot_decay_rates_576.py
python scripts/figures/aav/plot_data_validations.py

echo "=== Supplementary figures: AAV2 dataset ==="
python scripts/figures/aav/supp_plot_decay_rates.py
python scripts/figures/aav/supp_plot_mut_effs_post.py
python scripts/figures/aav/supp_plot_mut_effs_heatmaps.py

echo "=== Figure 4: Yeast dataset ==="
python scripts/figures/yeast/main_figure.py

echo "=== Supplementary figures: Yeast dataset ==="
python scripts/figures/yeast/supp_compare_mut_eff.py