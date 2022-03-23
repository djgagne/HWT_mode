import argparse
import yaml
from os.path import exists, join
from os import makedirs
import pandas as pd
import joblib
from hwtmode.data import load_patch_files, combine_patch_data, min_max_scale, get_meta_scalars, predict_labels_gmm, \
    predict_labels_cnn, predict_labels_dnn, save_labels, merge_labels, save_gridded_labels, load_labels
from hwtmode.process import get_neighborhood_probabilities
from hwtmode.models import load_conv_net
from hwtmode.evaluation import bss, brier_score
from tensorflow.keras.models import load_model


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the config file.")
    parser.add_argument("-e", "--eval", action="store_true", help="Evaluate conditional probabilities.")
    args = parser.parse_args()
    if not exists(args.config):
        raise FileNotFoundError(args.config + " not found.")
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    for dir in ['labels', 'evaluation']:
        makedirs(join(config["output_path"], dir), exist_ok=True)
    models, gmms, neuron_activations, labels = {}, {}, {}, {}
    if config["run_start_date"] == "today":
        if config['run_freq'] == 'hourly':
            start_str = (pd.Timestamp(config["run_start_date"], tz="UTC") - pd.Timedelta(hours=3)).strftime("%Y%m%d-%H00")
            end_str = (pd.Timestamp(config["run_end_date"], tz="UTC") - pd.Timedelta(hours=3)).strftime("%Y%m%d-%H00")
        elif config['run_freq'] == 'daily':
            start_str = (pd.Timestamp(config["run_start_date"], tz="UTC")).strftime("%Y%m%d-0000")
            end_str = (pd.Timestamp(config["run_end_date"], tz="UTC")).strftime("%Y%m%d-0000")
    else:
        start_str = (pd.Timestamp(config["run_start_date"], tz="UTC")).strftime("%Y%m%d-%H00")
        end_str = (pd.Timestamp(config["run_end_date"], tz="UTC")).strftime("%Y%m%d-%H00")

    l = []
    for d in pd.date_range(start_str.replace('-', ''), end_str.replace('-', ''), freq=config['run_freq'][0]):
        file_path = join(config["data_path"].replace('_nc', '_csv'),
                         f'{config["csv_model_prefix"]}{d.strftime("%Y%m%d-%H00")}.csv')
        print(file_path)
        if exists(file_path):
            df = pd.read_csv(file_path)
            l.append(df)

    storm_data = pd.concat(l).reset_index(drop=True)

    for model_type, model_dict in config["models"].items():
        for model_name in model_dict.keys():
            model_path = join(model_dict[model_name]["model_path"], "models", model_name)

            if model_type != 'supervised_DNN':
                scale_values = pd.read_csv(join(model_path, f"scale_values_{model_name}.csv"))
                scale_values['variable'] = model_dict[model_name]['input_variables']
                scale_values = scale_values.set_index('variable')

                print('Loading storm patches...')
                input_data, output, meta = load_patch_files(config["run_start_date"],
                                                            config["run_end_date"],
                                                            config["run_freq"],
                                                            config["data_path"],
                                                            model_dict[model_name]["input_variables"],
                                                            model_dict[model_name]["output_variables"],
                                                            config["patch_meta_variables"],
                                                            model_dict[model_name]["patch_radius"])

                input_combined = combine_patch_data(input_data, model_dict[model_name]["input_variables"])
                input_scaled, scale_values = min_max_scale(input_combined, scale_values)
                meta_df = get_meta_scalars(meta)
                models[model_name] = load_conv_net(model_path, model_name)
                print(model_name, f'({model_type})')
                print(models[model_name].model_.summary())

                if model_type == 'semi_supervised':

                    neuron_columns = [f"neuron_{n:03d}" for n in range(models[model_name].dense_neurons)]
                    neuron_activations[model_name] = pd.merge(meta_df, pd.DataFrame(0, columns=neuron_columns,
                                                              index=meta_df.index), left_index=True, right_index=True)
                    neuron_activations[model_name].loc[:, neuron_columns] = \
                        models[model_name].output_hidden_layer(input_scaled.values)

                    gmms[model_name] = joblib.load(join(f"{model_path}_{model_dict[model_name]['gmm_name']}.mod"))
                    cluster_assignments = joblib.load(join(model_path, f'{model_dict[model_name]["label_dict"]}'))

                    labels[model_name] = predict_labels_gmm(neuron_activations[model_name], gmms[model_name],
                                                            model_name, cluster_assignments)

                elif model_type == 'supervised':

                    labels[model_name] = predict_labels_cnn(input_scaled, models[model_name], model_name)

            elif model_type == 'supervised_DNN':
                models[model_name] = load_model(join(model_dict[model_name]['model_path'], f"{model_name}.h5"),
                                                custom_objects={"brier_score": brier_score, "brier_skill_score": bss})
                print(model_name, f'({model_type})')
                print(models[model_name].summary())
                with open(join(model_dict[model_name]["model_path"], 'scale_values.yaml'), "r") as config_file:
                    scale_values = yaml.load(config_file, Loader=yaml.Loader)
                labels[model_name] = predict_labels_dnn(storm_data, scale_values, models[model_name],
                                                        model_dict[model_name]["input_variables"],
                                                        model_name)

    all_labels = merge_labels(labels, storm_data, config["csv_meta_variables"], config["storm_variables"])
    save_labels(labels=all_labels,
                out_path=join(config['output_path'], "labels"),
                file_format=config['output_format'])

    if args.eval:

        model_names = []
        for model_class in config['models'].values():
            for model in model_class.keys():
                model_names.append(model)

        labels = load_labels(start=config["run_start_date"],
                             end=config["run_end_date"],
                             label_path=join(config["output_path"], "labels"),
                             run_freq=config["run_freq"],
                             file_format=config["output_format"])

        nprobs = get_neighborhood_probabilities(labels=labels,
                                                model_grid_path=config["model_grid_path"],
                                                model_names=model_names,
                                                proj_str=config["proj_str"])

        save_gridded_labels(ds=nprobs,
                            base_path=join(config["output_path"], "evaluation"),
                            tabular_format=config["output_format"])

    return

if __name__ == "__main__":
    main()
