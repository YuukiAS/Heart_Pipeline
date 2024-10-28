import os
from configuration import config


def make_experiment():
    experiment_path = config.output_path + config.experiment_name
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)

    model_path = experiment_path + "/" + config.model_name
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        weight = model_path + "/weights"
        os.mkdir(weight)
    print("[INFO] experiment directory has been created.")

    prediction_path = model_path + "/predictions"
    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)
