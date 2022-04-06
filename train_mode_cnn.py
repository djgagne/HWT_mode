import os
from hwtmode.data import load_patch_files, combine_patch_data, min_max_scale, storm_max_value, get_meta_scalars,\
    get_storm_variables
from hwtmode.models import BaseConvNet, load_conv_net
from hwtmode.evaluation import classifier_metrics
from hwtmode.interpretation import score_neurons, plot_neuron_composites, plot_saliency_composites, \
    plot_top_activations, cape_shear_modes, spatial_neuron_activations, \
    diurnal_neuron_activations, plot_prob_dist, plot_prob_cdf, plot_storm_clusters
from sklearn.mixture import GaussianMixture
import argparse
import yaml
from os.path import exists, join
from os import makedirs
import numpy as np
import tensorflow as tf
import xarray as xr
import joblib
import pandas as pd
import random


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the config file.")
    parser.add_argument("-t", "--train", action="store_true", help="Run neural network training.")
    parser.add_argument("-i", "--interp", action="store_true", help="Run interpretation.")
    parser.add_argument("-u", "--train_gmm", action="store_true", help="Run unsupervised model training.")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot interpretation results.")
    parser.add_argument("-p2", "--plot2", action="store_true", help="Plot additional interpretation results.")
    args = parser.parse_args()
    if not exists(args.config):
        raise FileNotFoundError(args.config + " not found.")
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
    np.random.seed(config["random_seed"])
    random.seed(config["random_seed"])
    tf.random.set_seed(config["random_seed"])
    # Load training data
    print(f"Loading training data period: {config['train_start_date']} to {config['train_end_date']}")
    data_input = {}
    output = {}
    out_max = {}
    storm_out = {}
    labels_combined = {}
    meta = {}
    meta_df = {}
    input_combined = {}
    input_scaled = {}
    scale_values = {}
    predictions = {}
    if exists('mask') in config:
        mask = config['mask']
    else:
        mask = False
    modes = ["train", "val", "test"]
    # Load training, validation, and testing data
    for mode in modes:
        data_input[mode], output[mode], meta[mode] = load_patch_files(config[mode + "_start_date"],
                                                                      config[mode + "_end_date"],
                                                                      None,
                                                                      config["data_path"],
                                                                      config["input_variables"],
                                                                      config["patch_output_variables"],
                                                                      config["meta_variables"],
                                                                      config["patch_radius"],
                                                                      mask)
        input_combined[mode] = combine_patch_data(data_input[mode], config["input_variables"])
        if mode == "train":
            input_scaled[mode], scale_values[mode] = min_max_scale(input_combined[mode])
        else:
            input_scaled[mode], scale_values[mode] = min_max_scale(input_combined[mode], scale_values["train"])
        out_max[mode] = storm_max_value(output[mode][config["patch_output_variables"][0]], meta[mode]["masks"])
        meta_df[mode] = get_meta_scalars(meta[mode])

        storm_out[mode] = get_storm_variables(start=config[mode + "_start_date"],
                                              end=config[mode + "_end_date"],
                                              data_path=config["csv_data_path"],
                                              csv_prefix=config["csv_prefix"],
                                              storm_vars=config["storm_output_variables"])

        labels_combined[mode] = np.concatenate([out_max[mode].reshape(-1, 1), storm_out[mode]], axis=1)

        if config["classifier"]:
            for i, val in enumerate(config["classifier_thresholds"]):
                labels_combined[mode][:, i] = np.where(labels_combined[mode][:, i] >
                                                       config["classifier_thresholds"][i], 1, 0)
        else:
            labels_combined[mode] = labels_combined[mode]
    del data_input, out_max
    for folder in ['models', 'plots', 'data', 'metrics', 'labels']:
        makedirs(join(config["out_path"], folder), exist_ok=True)
    with open(join(config['out_path'], 'full_config.yml'), "w") as config_file:
        yaml.dump(config, config_file)
    if "get_visible_devices" in dir(tf.config.experimental):
        gpus = tf.config.experimental.get_visible_devices("GPU")
        print("GPUS:", gpus)
    else:
        gpus = tf.config.get_visible_devices("GPU")
        print("GPUSS:", gpus)
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
            model_out_path = join(config["out_path"], "models", model_name)
            if not exists(model_out_path):
                makedirs(model_out_path)
            scale_values["train"].to_csv(join(model_out_path, "scale_values_" + model_name + ".csv"),
                                         index_label="variable")
            models[model_name] = BaseConvNet(**model_config)
            models[model_name].fit(input_scaled["train"].values, labels_combined["train"],
                                   val_x=input_scaled["val"].values, val_y=labels_combined["val"])
            models[model_name].save_model(model_out_path, model_name)
            for mode in modes:
                predictions[mode].loc[:, model_name] = models[model_name].predict(input_scaled[mode].values)[:, 0]
        for mode in modes:
            predictions[mode].to_csv(
                join(config["out_path"], "metrics", f"predictions_{mode}.csv"), index_label="index")

        print("Calculate metrics")
        # if config["classifier"]:
        #     model_scores = classifier_metrics(labels["test"], predictions["test"][list(config["models"].keys())])
        #     model_scores.to_csv(join(config["out_path"], "metrics", "model_test_scores.csv"), index_label="model_name")
    if args.interp:
        for model_name, model_config in config["models"].items():
            if model_name not in models.keys():
                model_out_path = join(config["out_path"], "models", model_name)
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
                neuron_activations[model_name][mode].to_csv(join(config["out_path"], "data",
                                                                 f"neuron_activations_{model_name}_{mode}.csv"),
                                                            index_label="index")
            #     saliency[model_name][mode] = models[model_name].saliency(input_scaled[mode])
            #
            #     saliency[model_name][mode].to_netcdf(join(config["out_path"], "data",
            #                                               f"neuron_saliency_{model_name}_{mode}.nc"),
            #                                          encoding={"saliency": {"zlib": True,
            #                                                                 "complevel": 4,
            #                                                                 "shuffle": True,
            #                                                                 "least_significant_digit": 3}})
            #     if config["classifier"]:
            #         neuron_scores[model_name].loc[mode] = score_neurons(labels[mode],
            #                                                             neuron_activations[model_name][mode][
            #                                                                 neuron_columns].values)
            #     else:
            #         neuron_scores[model_name].loc[mode] = score_neurons(labels[mode],
            #                                                             neuron_activations[model_name][mode][
            #                                                                 neuron_columns].values,
            #                                                             metric="r")
            #     del saliency[model_name][mode]
            # neuron_scores[model_name].to_csv(join(config["out_path"], "metrics",
            #                                       f"neuron_scores_{model_name}.csv"), index_label="mode")
            # del models[model_name], neuron_activations[model_name]

    if args.train_gmm:
        print('Begin Training Gaussian Mixture Model(s)')
        cluster_df = {}
        GMM = {}
        for model_name, model_config in config["models"].items():
            for mode in modes:
                neuron_activations[model_name] = {}
                neuron_activations[model_name][mode] = pd.read_csv(join(config["out_path"], "data",
                                                                        f"neuron_activations_{model_name}_{mode}.csv"))
                X = neuron_activations[model_name][mode].loc[
                    :, neuron_activations[model_name][mode].columns.str.contains('neuron')]
                for GMM_mod_name, GMM_config in config["GMM_models"].items():
                    if mode == "train":
                        GMM[GMM_mod_name] = GaussianMixture(**GMM_config).fit(X)
                    cluster_df[GMM_mod_name] = {}
                    cluster_df[GMM_mod_name][mode] = pd.DataFrame(GMM[GMM_mod_name].predict_proba(X),
                                                                  columns=[f"cluster {i}" for i in range(
                                                                      GMM_config['n_components'])])
                    cluster_df[GMM_mod_name][mode]['label prob'] = cluster_df[GMM_mod_name][mode].max(axis=1)
                    cluster_df[GMM_mod_name][mode]['label'] = GMM[GMM_mod_name].predict(X)
                    neuron_activations[model_name][mode].merge(
                        cluster_df[GMM_mod_name][mode], right_index=True, left_index=True).to_csv(join(
                        config["out_path"], "data", f"{model_name}_{GMM_mod_name}_{mode}_clusters.csv"), index=False)
                    joblib.dump(GMM[GMM_mod_name], join(
                        config["out_path"], "models", f'{model_name}_{GMM_mod_name}.mod'))

    if args.plot:
        print("Begin plotting")
        if "plot_kwargs" not in config.keys():
            config["plot_kwargs"] = {}
        for model_name, model_config in config["models"].items():
            print(model_name)
            if model_name not in models.keys():
                model_out_path = join(config["out_path"], "models", model_name)
                models[model_name] = load_conv_net(model_out_path, model_name)
                neuron_activations[model_name] = {}
                neuron_scores[model_name] = pd.read_csv(join(config["out_path"], "metrics",
                                                             f"neuron_scores_{model_name}.csv"), index_col="mode")
                saliency[model_name] = {}
            for mode in modes:
                print(mode)
                if mode not in neuron_activations[model_name].keys():
                    neuron_activations[model_name][mode] = pd.read_csv(join(config["out_path"], "data",
                                                                            f"neuron_activations_{model_name}_{mode}.csv"),
                                                                       index_col="index")
                    saliency[model_name][mode] = xr.open_dataarray(join(config["out_path"], "data",
                                                                        f"neuron_saliency_{model_name}_{mode}.nc"))
                for variable_name in config["input_variables"]:
                    print(variable_name)
                    if variable_name not in config["plot_kwargs"].keys():
                        plot_kwargs = None
                    else:
                        plot_kwargs = config["plot_kwargs"][variable_name]
                    plot_out_path = join(config["out_path"], "plots")
                    plot_neuron_composites(plot_out_path,
                                           model_name + "_" + mode,
                                           input_combined[mode],
                                           neuron_activations[model_name][mode].values,
                                           neuron_scores[model_name].loc[mode].values,
                                           variable_name, plot_kwargs=plot_kwargs)
                    plot_saliency_composites(plot_out_path,
                                             model_name + "_" + mode,
                                             saliency[model_name][mode], neuron_activations[model_name][mode].values,
                                             neuron_scores[model_name].loc[mode].values,
                                             variable_name)
                    plot_top_activations(plot_out_path,
                                         model_name + "_" + mode,
                                         input_combined[mode], meta_df[mode],
                                         neuron_activations[model_name][mode],
                                         neuron_scores[model_name].loc[mode].values,
                                         saliency[model_name][mode],
                                         variable_name, plot_kwargs=plot_kwargs)
                del saliency[model_name][mode]
    if args.plot2:
        print("Additional Plotting...")
        for model_name in config["models"].keys():
            for mode in ["val"]:
                for GMM_mod_name, GMM_config in config["GMM_models"].items():
                    plot_out_path = join(config["out_path"], "plots", model_name, GMM_mod_name)
                    if not exists(plot_out_path):
                        makedirs(plot_out_path, exist_ok=True)
                    cluster_df = pd.read_csv(join(
                        config["out_path"], "data", f"{model_name}_{GMM_mod_name}_{mode}_clusters.csv"))
                    plot_prob_dist(cluster_df, plot_out_path, GMM_mod_name, GMM_config["n_components"])
                    plot_prob_cdf(cluster_df, plot_out_path, GMM_mod_name, GMM_config["n_components"])
                    cape_shear_modes(cluster_df, plot_out_path, config["data_path"], mode, model_name,
                                     gmm_name=GMM_mod_name, cluster=True, num_storms=1000)
                    spatial_neuron_activations(cluster_df, plot_out_path, mode, model_name,
                                               gmm_name=GMM_mod_name, cluster=True)
                    diurnal_neuron_activations(cluster_df, plot_out_path, mode, model_name,
                                               gmm_name=GMM_mod_name, cluster=True)
                    for prob_type in ['highest', 'lowest']:
                        plot_storm_clusters(config['data_path'], plot_out_path, cluster_df,
                                            n_storms=25,
                                            patch_radius=config["patch_radius"],
                                            prob_type=prob_type,
                                            seed=config["random_seed"])

    return


if __name__ == "__main__":
    main()
