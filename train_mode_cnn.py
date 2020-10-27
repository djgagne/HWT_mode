from memory_profiler import profile
from hwtmode.data import load_patch_files, combine_patch_data, min_max_scale, storm_max_value, get_meta_scalars
from hwtmode.models import BaseConvNet, load_conv_net
from hwtmode.evaluation import classifier_metrics
from hwtmode.interpretation import score_neurons, plot_neuron_composites, plot_saliency_composites, plot_top_activations
import argparse
import yaml
from os.path import exists, join
from os import makedirs
import numpy as np
import tensorflow as tf
import xarray as xr
import pandas as pd


fp=open('memory_profiler.log','w+')
@profile(precision=4, stream=fp)
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the config file.")
    parser.add_argument("-t", "--train", action="store_true", help="Run neural network training.")
    parser.add_argument("-i", "--interp", action="store_true", help="Run interpretation.")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot interpretation results.")
    args = parser.parse_args()
    if not exists(args.config):
        raise FileNotFoundError(args.config + " not found.")
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
    # Load training data
    print(f"Loading training data period: {config['train_start_date']} to {config['train_end_date']}")
    data_input = {}
    output = {}
    out_max = {}
    labels = {}
    meta = {}
    meta_df = {}
    input_combined = {}
    input_scaled = {}
    scale_values = {}
    predictions = {}
    modes = ["train", "val", "test"]
    # Load training, validation, and testing data
    for mode in modes:
        data_input[mode], output[mode], meta[mode] = load_patch_files(config[mode + "_start_date"],
                                                                 config[mode + "_end_date"],
                                                                 config["data_path"],
                                                                 config["input_variables"],
                                                                 config["output_variables"],
                                                                 config["meta_variables"],
                                                                 config["patch_radius"])
        input_combined[mode] = combine_patch_data(data_input[mode], config["input_variables"])
        if mode == "train":
            input_scaled[mode], scale_values[mode] = min_max_scale(input_combined[mode])
        else:
            input_scaled[mode], scale_values[mode] = min_max_scale(input_combined[mode], scale_values["train"])
        out_max[mode] = storm_max_value(output[mode][config["output_variables"][0]], meta[mode]["masks"])
        meta_df[mode] = get_meta_scalars(meta[mode])
        print(meta_df[mode].columns)
        if config["classifier"]:
            labels[mode] = np.where(out_max[mode] >= config["classifier_threshold"], 1, 0)
        else:
            labels[mode] = out_max[mode]
    if not exists(config["out_path"]):
        makedirs(config["out_path"])
    scale_values["train"].to_csv(join(config["out_path"], "scale_values.csv"),
                                 index_label="variable")
    if "get_visible_devices" in dir(tf.config.experimental):
        gpus = tf.config.experimental.get_visible_devices("GPU")
    else:
        gpus = tf.config.get_visible_devices("GPU")
    for device in gpus:
        tf.config.experimental.set_memory_growth(device, True)
    models = {}
    neuron_activations = {}
    neuron_scores = {}
    saliency = {}
    if args.train:
        print("Begin model training")
        for mode in modes:
            predictions[mode] = pd.DataFrame(0, index=meta_df[mode].index,
                                        columns=list(config["models"].keys()))
            predictions[mode] = pd.merge(meta_df[mode], predictions[mode], left_index=True, right_index=True)
        for model_name, model_config in config["models"].items():
            model_out_path = join(config["out_path"], model_name)
            if not exists(model_out_path):
                makedirs(model_out_path)
            scale_values["train"].to_csv(join(model_out_path, "scale_values_" + model_name + ".csv"),
                                      index_label="variable")
            models[model_name] = BaseConvNet(**model_config)
            models[model_name].fit(input_scaled["train"].values, labels["train"],
                                   val_x=input_scaled["val"].values, val_y=labels["val"])
            models[model_name].save_model(model_out_path, model_name)
            for mode in modes:
                predictions[mode].loc[:, model_name] = models[model_name].predict(input_scaled[mode].values)
        for mode in modes:
            predictions[mode].to_csv(join(config["out_path"], f"predictions_{mode}.csv"), index_label="index")
        print("Calculate metrics")
        if config["classifier"]:
            model_scores = classifier_metrics(labels["test"], predictions["test"][list(config["models"].keys())])
            model_scores.to_csv(join(config["out_path"], "model_test_scores.csv"), index_label="model_name")
    if args.interp:
        for model_name, model_config in config["models"].items():
            if model_name not in models.keys():
                model_out_path = join(config["out_path"], model_name)
                models[model_name] = load_conv_net(model_out_path, model_name)
            neuron_columns = [f"neuron_{n:03d}" for n in range(models[model_name].dense_neurons)]
            neuron_activations[model_name] = {}
            neuron_scores[model_name] = pd.DataFrame(0, columns=neuron_columns, index=modes)
            saliency[model_name] = {}
            for mode in modes:
                neuron_activations[model_name][mode] = pd.merge(meta_df[mode], pd.DataFrame(0, columns=neuron_columns,
                                                                          index=meta_df[mode].index),
                                                    left_index=True, right_index=True)
                neuron_activations[model_name][mode].loc[:, neuron_columns] = models[model_name].output_hidden_layer(
                    input_scaled[mode].values)
                neuron_activations[model_name][mode].to_csv(join(config["out_path"],
                                                     f"neuron_activations_{model_name}_{mode}.csv"),
                                                index_label="index")
                saliency[model_name][mode] = models[model_name].saliency(input_scaled[mode])

                saliency[model_name][mode].to_netcdf(join(config["out_path"],
                                                          f"neuron_saliency_{model_name}_{mode}.nc"),
                                                     encoding={"saliency": {"zlib": True,
                                                                            "complevel": 4,
                                                                            "shuffle": True,
                                                                            "least_significant_digit": 3}})
                if config["classifier"]:
                    neuron_scores[model_name].loc[mode] = score_neurons(labels[mode],
                                                                        neuron_activations[model_name][mode][neuron_columns].values)
                else:
                    neuron_scores[model_name].loc[mode] = score_neurons(labels[mode],
                                                                        neuron_activations[model_name][mode][neuron_columns].values,
                                                        metric="r")
            neuron_scores[model_name].to_csv(join(config["out_path"],
                                             f"neuron_scores_{model_name}.csv"), index_label="mode")
    if args.plot:
        print("Begin plotting")
        if "plot_kwargs" not in config.keys():
            config["plot_kwargs"] = {}
        for model_name, model_config in config["models"].items():
            print(model_name)
            if model_name not in models.keys():
                model_out_path = join(config["out_path"], model_name)
                models[model_name] = load_conv_net(model_out_path, model_name)
                neuron_activations[model_name] = {}
                neuron_scores[model_name] = pd.read_csv(join(config["out_path"],
                                                        f"neuron_scores_{model_name}.csv"), index_col="mode")
                saliency[model_name] = {}
            for mode in modes:
                print(mode)
                if mode not in neuron_activations[model_name].keys():
                    neuron_activations[model_name][mode] = pd.read_csv(join(config["out_path"],
                                                           f"neuron_activations_{model_name}_{mode}.csv"),
                                                           index_col="index")
                    saliency[model_name][mode] = xr.open_dataarray(join(config["out_path"],
                                                                        f"neuron_saliency_{model_name}_{mode}.nc"))
                for variable_name in config["input_variables"]:
                    print(variable_name)
                    if variable_name not in config["plot_kwargs"].keys():
                        plot_kwargs = None
                    else:
                        plot_kwargs = config["plot_kwargs"][variable_name]
                    plot_neuron_composites(config["out_path"], model_name + "_" + mode,
                                           input_combined[mode],
                                           neuron_activations[model_name][mode].values,
                                           neuron_scores[model_name].loc[mode].values,
                                           variable_name, plot_kwargs=plot_kwargs)
                    plot_saliency_composites(config["out_path"], model_name + "_" + mode,
                                             saliency[model_name][mode], neuron_activations[model_name][mode].values,
                                             neuron_scores[model_name].loc[mode].values,
                                             variable_name)
                    plot_top_activations(config["out_path"], model_name + "_" + mode,
                                         input_combined[mode], meta_df[mode],
                                         neuron_activations[model_name][mode],
                                         neuron_scores[model_name].loc[mode].values,
                                         saliency[model_name][mode],
                                         variable_name, plot_kwargs=plot_kwargs)
    return


if __name__ == "__main__":
    main()
