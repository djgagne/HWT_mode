from hwtmode.data import load_patch_files, combine_patch_data, min_max_scale, storm_max_value
from hwtmode.models import BaseConvNet, load_conv_net
from hwtmode.evaluation import classifier_metrics
from hwtmode.interpretation import score_neurons, plot_neuron_composites, plot_saliency_composites
import argparse
import yaml
from os.path import exists, join
from os import makedirs
import numpy as np
import tensorflow as tf
import pandas as pd


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the config file.")
    parser.add_argument("-t", "--train", action="store_true", help="Run neural network training.")
    parser.add_argument("-i", "--interp", action="store_true", help="Run interpretation.")
    args = parser.parse_args()
    if not exists(args.config):
        raise FileNotFoundError(args.config + " not found.")
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
    # Load training data
    print(f"Loading training data period: {config['train_start_date']} to {config['train_end_date']}")
    train_input, train_output, train_meta = load_patch_files(config["train_start_date"],
                                                             config["train_end_date"],
                                                             config["data_path"],
                                                             config["input_variables"],
                                                             config["output_variables"],
                                                             config["meta_variables"],
                                                             config["patch_radius"])
    train_input_combined = combine_patch_data(train_input, config["input_variables"])
    train_input_scaled, train_scale_values = min_max_scale(train_input_combined)
    train_out_max = storm_max_value(train_output[config["output_variables"][0]], train_meta["masks"])

    # Load validation data
    val_input, val_output, val_meta = load_patch_files(config["val_start_date"],
                                                       config["val_end_date"],
                                                       config["data_path"],
                                                       config["input_variables"],
                                                       config["output_variables"],
                                                       config["meta_variables"],
                                                       config["patch_radius"])
    val_input_combined = combine_patch_data(val_input, config["input_variables"])
    val_input_scaled, val_scale_values = min_max_scale(val_input_combined,
                                                       scale_values=train_scale_values)
    val_out_max = storm_max_value(val_output[config["output_variables"][0]], val_meta["masks"])
    # Load testing data
    test_input, test_output, test_meta = load_patch_files(config["test_start_date"],
                                                          config["test_end_date"],
                                                          config["data_path"],
                                                          config["input_variables"],
                                                          config["output_variables"],
                                                          config["meta_variables"],
                                                          config["patch_radius"])
    test_input_combined = combine_patch_data(test_input, config["input_variables"])
    test_input_scaled, test_scale_values = min_max_scale(test_input_combined,
                                                         scale_values=train_scale_values)
    test_out_max = storm_max_value(test_output[config["output_variables"][0]], test_meta["masks"])
    if config["classifier"]:
        train_labels = np.where(train_out_max >= config["classifier_threshold"], 1, 0)
        val_labels = np.where(val_out_max >= config["classifier_threshold"], 1, 0)
        test_labels = np.where(test_out_max >= config["classifier_threshold"], 1, 0)
    else:
        train_labels = train_out_max
        val_labels = val_out_max
        test_labels = test_out_max
    if not exists(config["out_path"]):
        makedirs(config["out_path"])
    if "get_visible_devices" in dir(tf.config.experimental):
        gpus = tf.config.experimental.get_visible_devices("GPU")
    else:
        gpus = tf.config.get_visible_devices("GPU")
    for device in gpus:
        tf.config.experimental.set_memory_growth(device, True)
    models = {}
    if args.train:
        test_predictions = pd.DataFrame(0, index=np.arange(test_labels.size),
                                        columns=list(config["models"].keys()))
        for model_name, model_config in config["models"].items():
            model_out_path = join(config["out_path"], model_name)
            if not exists(model_out_path):
                makedirs(model_out_path)
            train_scale_values.to_csv(join(model_out_path, "scale_values_ " + model_name + ".csv"),
                                      index_label="variable")
            models[model_name] = BaseConvNet(**model_config)
            models[model_name].fit(train_input_scaled.values, train_labels,
                                   val_x=val_input_scaled.values, val_y=val_labels)
            test_predictions.loc[:, model_name] = models[model_name].predict(test_input_scaled.values)
            models[model_name].save_model(model_out_path, model_name)
        model_scores = classifier_metrics(test_labels, test_predictions)
        model_scores.to_csv(join(config["out_path"], "model_test_scores.csv"), index_label="model_name")
    if args.interp:
        for model_name, model_config in config["models"].items():
            if model_name not in models.keys():
                model_out_path = join(config["out_path"], model_name)
                models[model_name] = load_conv_net(model_out_path, model_name)
            train_neuron_activations = models[model_name].output_hidden_layer(train_input_scaled.values)
            train_saliency = models[model_name].saliency(train_input_scaled.values)
            if config["classifier"]:
                train_neuron_scores = score_neurons(train_labels, train_neuron_activations)
            else:
                train_neuron_scores = score_neurons(train_labels, train_neuron_activations, metric="r")
            plot_neuron_composites(config["out_path"], model_name, train_input_combined, train_neuron_activations,
                                   train_neuron_scores)
            plot_saliency_composites(config["out_path"], model_name, train_saliency, train_neuron_activations,
                                     train_neuron_scores)



    return


if __name__ == "__main__":
    main()
