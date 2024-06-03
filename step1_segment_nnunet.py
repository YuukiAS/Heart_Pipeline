nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD
nnUNetv2_train DATASET_NAME_OR_ID 3d_lowres FOLD
nnUNetv2_train DATASET_NAME_OR_ID 3d_cascade_fullres FOLD

nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS 
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
nnUNetv2_apply_postprocessing -i FOLDER_WITH_PREDICTIONS -o OUTPUT_FOLDER --pp_pkl_file POSTPROCESSING_FILE -plans_json PLANS_FILE -dataset_json DATASET_JSON_FILE
