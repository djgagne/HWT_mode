from hwtmode.interpretation import plot_cluster_dist, plot_prob_dist, plot_prob_cdf, plot_storm_clusters
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from os.path import join, exists
from os import makedirs
import yaml
import joblib
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the config file.")
    args = parser.parse_args()

    if not exists(args.config):
        raise FileNotFoundError(args.config + " not found.")

    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    patch_data_path = config['patch_data_path']
    base_neuron_act_path = config['base_neuron_act_path']
    train_activations = config['train_activations']
    val_activations = config['val_activations']
    test_activations = config['test_activations']
    output_path = config['output_path']
    cluster_types = config['cluster_types']
    n_clusters = config['n_clusters']
    n_samps = config['n_samps']
    plot_mode = config['plot_mode']
    seed = config['seed']

    if not exists(output_path):
        makedirs(output_path)

    train = pd.read_csv(join(base_neuron_act_path, train_activations))

    if plot_mode == 'train':
        plot_data = train
    elif plot_mode == 'val':
        plot_data = pd.read_csv(join(base_neuron_act_path, val_activations))
    elif plot_mode == 'test':
        plot_data = pd.read_csv(join(base_neuron_act_path, test_activations))


    for cluster_type in cluster_types:

        for n_cluster in n_clusters:

            if cluster_type == 'GMM':

                X = train.loc[:, train.columns.str.contains('neuron')]
                mod = GaussianMixture(n_components=n_cluster, **config['GMM_kwargs']).fit(X)
                max_cluster_prob = mod.predict_proba(
                    plot_data.loc[:, plot_data.columns.str.contains('neuron')]).max(axis=1)
                plot_data['cluster_prob'] = max_cluster_prob
                plot_data['cluster'] = mod.predict(plot_data.loc[:, plot_data.columns.str.contains('neuron')])
                plot_prob_dist(plot_data, output_path, cluster_type, n_cluster)
                plot_prob_cdf(plot_data, output_path, cluster_type, n_cluster)

            elif cluster_type == 'Spectral':

                X = train.sample(n_samps, random_state=seed)
                X_train = X.loc[:, X.columns.str.contains('neuron')]
                mod = SpectralClustering(n_components=n_cluster, **config["Spectral_kwargs"]).fit(X_train)
                plot_data = X
                plot_data['cluster'] = mod.labels_

            joblib.dump(mod, join(output_path, f'{cluster_type}_{n_cluster}_clusters.mod'))

            plot_cluster_dist(plot_data, output_path, cluster_type, n_cluster)
            plot_storm_clusters(patch_data_path, output_path, plot_data, cluster_type, seed, **config['plot_kwargs'])

    return

if __name__ == "__main__":
    main()