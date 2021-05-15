# Odaka_DNN_inverse_2011_tsunami
# Version 1.0
This is a code for performing inverse analysis of tsunami deposits using deep-learning neural network. The forward model fittnuss produces datasets of the thickness distribution of tsunami deposits with random initial conditions, and DNN constructed with tensorflow and keras learns the relation between initial conditions and depositional features. Then, the trained DNN model works as the inverse model for ancient or modern tsunami deposits. See details in Mitra et al., (2020) and Naruse and Abe (2017).  

Explanation of files Version 1.0:

Forward_model_for_DNN_J2_odaka_GS_round2200.py the forward model for deposition from tsunamis

Odaka_SW1800_final_rev.ipynb: a jupyter notebook for performing the inversion

start_param_random_5000_j2_odaka_round_2200.csv: teacher data. Initial conditions used for production of training datasets.

eta_5000_g6_300grid_j2_odaka_round_2200.csv: training and test data produced by the forward model. This file is too large to store in GitHub, so that it is only available from Zenodo repository.

odaka_increased_class_edit3.csv: Dataset of 2011 Tohoku-oki tsunami measured at the Odaka region, Japan. Volume-per-unit-area of 5 grain size classes were recorded.

config_g6_300grid_j2_gs_round.ini: Configuration file of the forward model used for production of the training datasets and inversion.

GS_calculation.xlsx: Detailed calculation of measured grain-size distribution from Odaka region, Japan
