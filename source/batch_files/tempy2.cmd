python ..\comp_stat_vals.py NMB AD all --split_size=1
python ..\ext_raw_measures.py NMB all all
python ..\ft_sel_red.py NMB AD all pca --num_features=30 --no_normalize --batch