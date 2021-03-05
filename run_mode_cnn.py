import argparse
import yaml
from os.path import exists, join
from os import makedirs
import pandas as pd
import joblib
from hwtmode.data import load_patch_files, combine_patch_data, min_max_scale, get_meta_scalars, predict_labels
from hwtmode.models import load_conv_net


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the config file.")
    args = parser.parse_args()
    if not exists(args.config):
        raise FileNotFoundError(args.config + " not found.")
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
    input_data, output, meta = load_patch_files(config["run_start_date"],
                                                config["run_end_date"],
                                                config["data_path"],
                                                config["input_variables"],
                                                config["output_variables"],
                                                config["meta_variables"],
                                                config["patch_radius"])
    input_combined = combine_patch_data(input_data, config["input_variables"])
    print(list(meta.variables.keys()))
    meta_df = get_meta_scalars(meta)
    models, gmms, labels, neuron_activations = {}, {}, {}, {}
    if not exists(config["activation_path"]):
        makedirs(config["activation_path"])
    for model_name in config["models"]:

        print(model_name)
        scale_values = pd.read_csv(join(config["out_path"], model_name, f"scale_values_{model_name}.csv"),
                                   index_col="variable")
        input_scaled, scale_values = min_max_scale(input_combined, scale_values)
        print("Input shape", input_scaled.shape)
        model_out_path = join(config["out_path"], model_name)
        models[model_name] = load_conv_net(model_out_path, model_name)
        print(models[model_name].model_.summary())
        neuron_columns = [f"neuron_{n:03d}" for n in range(models[model_name].dense_neurons)]
        neuron_activations[model_name] = pd.merge(meta_df, pd.DataFrame(0, columns=neuron_columns, index=meta_df.index),
                                                  left_index=True, right_index=True)
        neuron_activations[model_name].loc[:, neuron_columns] = \
            models[model_name].output_hidden_layer(input_scaled.values)
        print("Neuron activation shape:", neuron_activations[model_name].shape)
        neuron_activations[model_name].to_csv(join(config["activation_path"], f'{model_name}_activations.csv'),
                                              index=False)
        # run_dates = pd.DatetimeIndex(neuron_activations[model_name]["run_date"].unique())
        # print(run_dates)

        gmms[model_name] = joblib.load(join(config["out_path"], f'{model_name}.gmm'))
        cluster_assignments = joblib.load(join(config["out_path"], f'{model_name}_gmm_labels.dict'))
        labels[model_name] = predict_labels(neuron_activations[model_name], neuron_columns, gmms[model_name],
                                            cluster_assignments)
        labels[model_name].to_csv(join(config["labels_path"], f'{model_name}_predictions.csv'), index=False)
        print("Completed.")

    return


if __name__ == "__main__":
    main()
