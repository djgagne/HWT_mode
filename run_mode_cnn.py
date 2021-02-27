import argparse
import yaml
from os.path import exists, join
from os import makedirs
import pandas as pd
import numpy as np
import joblib
from hwtmode.data import load_patch_files, combine_patch_data, min_max_scale, storm_max_value, get_meta_scalars
from hwtmode.models import load_conv_net
#from hwtmode.analysis import plot_storm_mode_analysis_map

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
    out_max = storm_max_value(output[config["output_variables"][0]], meta["masks"])
    print(list(meta.variables.keys()))
    meta_df = get_meta_scalars(meta)
    models = {}
    gmms = {}
    neuron_activations = {}
    if not exists(config["activation_path"]):
        makedirs(config["activation_path"])
    for model_name in config["models"]:
        print(model_name)
        scale_values = pd.read_csv(join(config["out_path"], model_name, f"scale_values_{model_name}.csv"), index_col="variable")

        input_scaled, scale_values = min_max_scale(input_combined, scale_values)
        print("Input shape", input_scaled.shape)
        model_out_path = join(config["out_path"], model_name)
        models[model_name] = load_conv_net(model_out_path, model_name)
        print(models[model_name].model_.summary())
        neuron_columns = [f"neuron_{n:03d}" for n in range(models[model_name].dense_neurons)]
        neuron_activations[model_name] = pd.merge(meta_df, pd.DataFrame(0, columns=neuron_columns,
                                                                          index=meta_df.index),
                                                    left_index=True, right_index=True)
        neuron_activations[model_name].loc[:, neuron_columns] = models[model_name].output_hidden_layer(input_scaled.values)
        print("Neuron activation shape:", neuron_activations[model_name].shape)
        run_dates = pd.DatetimeIndex(neuron_activations[model_name]["run_date"].unique())
        print(run_dates)

        gmms[model_name] = joblib.load(join(config["out_path"], f'{model_name}.gmm'))
        gmms[model_name].predict(neuron_activations[model_name].loc[:, neuron_columns])
        gmms[model_name].predict_proba(neuron_activations[model_name].loc[:, neuron_columns]).max(axis=1)
        labels = joblib.load(join(config["out_path"], f'{model_name}_gmm_labels.dict'))

        # for run_date in run_dates:
        #     print(run_date)
        #     rdi = neuron_activations[model_name]["run_date"] == run_date
        #     print("Run storm count", np.count_nonzero(rdi))
        #     run_date_str = run_date.strftime(config["date_format"])
        #     na_file = join(config["activation_path"],
        #                    f"neuron_activations_{model_name}_{run_date_str}.csv")
        #     neuron_activations[model_name].loc[rdi].to_csv(na_file, index_label="index")
        #     meta_run = meta.isel(p=np.where(rdi)[0])
        #     print(run_date.strftime("%Y-%m-%d") + " plot all storms")
            # plot_storm_mode_analysis_map(neuron_activations[model_name].loc[rdi],
            #                              meta_run, config["models"][model_name],
            #                              run_date, 1, 35, model_name, config["activation_path"], period_total=True)
            # plot_storm_mode_analysis_map(neuron_activations[model_name].loc[rdi],
            #                              meta_run, config["models"][model_name],
            #                              run_date, 12, 35, model_name, config["activation_path"], period_total=True)
            # print(run_date.strftime("%Y-%m-%d") + " plot hourly storms")
            # plot_storm_mode_analysis_map(neuron_activations[model_name].loc[rdi],
            #                              meta_run, config["models"][model_name],
            #                              run_date, config["start_hour"], config["end_hour"], model_name,
            #                              config["activation_path"],
            #                              period_total=False)

            ### LOAD GMM, make predictions, append predictions to CSV ###


    return


if __name__ == "__main__":
    main()
