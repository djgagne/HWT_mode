import argparse
import yaml
from os.path import exists, join
from os import makedirs
import pandas as pd
import numpy as np
from hwtmode.data import load_patch_files, combine_patch_data, min_max_scale, storm_max_value, get_meta_scalars
from hwtmode.models import load_conv_net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the config file.")
    parser.add_argument("-i", "--interp", action="store_true", help="Run interpretation.")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot interpretation results.")
    args = parser.parse_args()
    if not exists(args.config):
        raise FileNotFoundError(args.config + " not found.")
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
    input, output, meta = load_patch_files(config["run_start_date"],
                                           config["run_end_date"],
                                           config["data_path"],
                                           config["input_variables"],
                                           config["output_variables"],
                                           config["meta_variables"],
                                           config["patch_radius"])
    input_combined = combine_patch_data(input, config["input_variables"])
    scale_values = pd.read_csv(join(config["out_path"], "scale_values.csv"))
    input_scaled, scale_values = min_max_scale(input_combined, scale_values)
    out_max = storm_max_value(output[config["output_variables"][0]], meta["masks"])
    meta_df = get_meta_scalars(meta)
    if config["classifier"]:
        labels = np.where(out_max >= config["classifier_threshold"], 1, 0)
    else:
        labels = out_max
    models = {}
    neuron_activations = {}
    if not exists(config["activation_path"]):
        makedirs(config["activation_path"])
    for model_name in config["models"]:
        model_out_path = join(config["out_path"], model_name)
        models[model_name] = load_conv_net(model_out_path, model_name)
        neuron_columns = [f"neuron_{n:03d}" for n in range(models[model_name].dense_neurons)]
        neuron_activations[model_name] = pd.merge(meta_df, pd.DataFrame(0, columns=neuron_columns,
                                                                          index=meta_df.index),
                                                    left_index=True, right_index=True)
        neuron_activations[model_name].loc[:, neuron_columns] = models[model_name].output_hidden_layer(input_scaled)
        run_dates = neuron_activations[model_name]["run_date"].unique()
        for run_date in run_dates:
            rdi = neuron_activations[model_name]["run_date"] == run_date
            run_date_str = run_date.strftime(config["date_format"])
            na_file = join(config["activation_path"],
                           f"neuron_activations_{model_name}_{run_date_str}.csv")
            neuron_activations[model_name].loc[rdi].to_csv(na_file, index_col="index")
    return


if __name__ == "__main__":
    main()