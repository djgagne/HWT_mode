import argparse
import yaml
from os.path import exists, join
from os import makedirs
import pandas as pd
import numpy as np
import joblib
from hwtmode.data import load_patch_files, combine_patch_data, min_max_scale, get_meta_scalars, predict_labels_gmm, \
    predict_labels_cnn, get_contours
from hwtmode.models import load_conv_net
from hwtmode.analysis import plot_storm_mode_analysis_map


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the config file.")
    parser.add_argument("-p", "--plot_activation", action="store_true", help="Plot storms by specified active neurons.")
    args = parser.parse_args()
    if not exists(args.config):
        raise FileNotFoundError(args.config + " not found.")
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
    for path in [config["activation_path"], config["labels_path"]]:
        if not exists(path):
            makedirs(path)
    models, gmms, neuron_activations = {}, {}, {}

    for model_type, model_dict in config["models"].items():
        for model_name in model_dict.keys():

            print(model_name, f'({model_type})')
            scale_values = pd.read_csv(join(config["out_path"], model_name, f"scale_values_{model_name}.csv"),
                                       index_col="variable")
            input_data, output, meta = load_patch_files(config["run_start_date"],
                                                        config["run_end_date"],
                                                        config["data_path"],
                                                        model_dict[model_name]["input_variables"],
                                                        model_dict[model_name]["output_variables"],
                                                        config["meta_variables"],
                                                        model_dict[model_name]["patch_radius"])
            input_combined = combine_patch_data(input_data, model_dict[model_name]["input_variables"])
            input_scaled, scale_values = min_max_scale(input_combined, scale_values)
            meta_df = get_meta_scalars(meta)
            geometry_df = get_contours(meta)

            print("Input shape", input_scaled.shape)

            model_out_path = join(config["out_path"], model_name)
            models[model_name] = load_conv_net(model_out_path, model_name)
            print(models[model_name].model_.summary())

            if model_type == 'semi_supervised':

                neuron_columns = [f"neuron_{n:03d}" for n in range(models[model_name].dense_neurons)]
                neuron_activations[model_name] = pd.merge(meta_df, pd.DataFrame(0, columns=neuron_columns,
                                                          index=meta_df.index), left_index=True, right_index=True)
                neuron_activations[model_name].loc[:, neuron_columns] = \
                    models[model_name].output_hidden_layer(input_scaled.values)
                neuron_activations[model_name].to_csv(join(config["activation_path"], f'{model_name}_activations.csv'),
                                                      index=False)

                gmms[model_name] = joblib.load(join(config["out_path"], model_name, f'{model_name}.gmm'))
                cluster_assignments = joblib.load(join(config["out_path"], model_name, f'{model_name}_gmm_labels.dict'))
                labels = predict_labels_gmm(neuron_activations[model_name], neuron_columns, gmms[model_name],
                                        cluster_assignments)
                labels = pd.merge(labels, geometry_df)
                labels.insert(1, 'forecast_hour', ((labels['time'] - labels['run_date']) /
                                                    pd.Timedelta(hours=1)).astype('int32'))
            elif model_type == 'supervised':

                labels = predict_labels_cnn(input_scaled, geometry_df, models[model_name])

            labels.to_pickle(join(config["labels_path"], f'{model_name}_labels.pkl'))

    print("Completed.")

    return


if __name__ == "__main__":
    main()
