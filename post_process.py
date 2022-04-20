import yaml
import argparse
import os
import time
from os.path import join, exists
import pandas as pd
from hwtmode.process import fetch_storm_reports, generate_obs_grid, generate_mode_grid, load_HRRR_proxy_data,\
    load_WRF_proxy_data, get_quantiles, get_proxy_events


def main():

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Name of the config file.")
    args = parser.parse_args()
    if not exists(args.config):
        raise FileNotFoundError(args.config + " not found.")
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)

    start_date = config['start_date']
    end_date = config['end_date']
    label_path = config['label_path']
    model_grid_path = config['model_grid_path']
    output_path = config['output_path']
    HRRR_lat_lon_grid = config['HRRR_lat_lon_grid']
    AWS_bucket = config['HRRR_AWS_bucket']
    WRF_data_path = config['WRF_data_path']
    cnn = config['cnn']
    gmm = config['gmm']
    prob_bin_width = config['prob_bin_width']
    HRRR_proxy_vars = config['HRRR_proxy_variables']
    WRF_proxy_vars = config['WRF_proxy_variables']
    use_saved_indices = config['use_saved_indices']
    HRRR_indices_path = config['HRRR_indices_path']
    WRF_indices_path = config['WRF_indices_path']
    run_freq = config['run_freq']
    max_forecast_len = config['max_forecast_len']
    proxy_quantile_vals = config['proxy_quantile_vals']
    dated_output_path = output_path + pd.Timestamp("today").strftime('%Y%m%d')
    storm_report_path = join(dated_output_path, 'SPC_Storm_Reports')

    for path in [output_path, dated_output_path, storm_report_path]:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=False)

    for report_type in ['filtered_torn', 'filtered_hail', 'filtered_wind']:
        print(f'Downloading SPC storm reports from {start_date} through {end_date} for {report_type}')
        fetch_storm_reports(start_date, end_date, storm_report_path, report_type)

    data = {}
    for model in [cnn, gmm]:
        for physical_model in ['WRF', 'HRRR']:
            if physical_model == 'HRRR':
                labels = join(label_path, 'HRRR_rerun')
                proj_str = config["HRRR_proj_str"]
            else:
                labels = join(label_path, 'WRF')
                proj_str = config["WRF_proj_str"]

            print(f'Aggregating storm reports to a grid.')
            obs = generate_obs_grid(beg=start_date,
                                    end=end_date,
                                    storm_report_path=storm_report_path,
                                    model_grid_path=model_grid_path,
                                    proj_str=proj_str)

            print(f'Constructing gridded {physical_model} predictions at daily intervals for the {model} model.')
            data[f'{model}_{physical_model}_1d'] = generate_mode_grid(beg=start_date,
                                                                      end=end_date,
                                                                      labels=labels,
                                                                      model_grid_path=model_grid_path,
                                                                      min_lead_time=1,
                                                                      max_lead_time=max_forecast_len,
                                                                      proj_str=proj_str,
                                                                      run_date_freq=run_freq,
                                                                      bin_width=prob_bin_width)

    for k, v in data.items():
        if not (obs['valid_time'].values == v['valid_time'].values).sum() == len(obs['valid_time']):
            raise ValueError('Valid times do not match between SPC Storm Reports and predictions!')
        v.to_netcdf(join(dated_output_path, f'{k}_predictions.nc'))
    obs.to_netcdf(join(dated_output_path, f'gridded_storm_reports.nc'))
    #
    # HRRR_proxy_ds = load_HRRR_proxy_data(beg=start_date,
    #                                      end=end_date,
    #                                      freq=run_freq,
    #                                      variables=HRRR_proxy_vars,
    #                                      max_forecast_len=max_forecast_len,
    #                                      AWS_bucket=AWS_bucket,
    #                                      HRRR_model_map=HRRR_lat_lon_grid)
    #
    # if not (obs['valid_time'].values == HRRR_proxy_ds['valid_time'].values).sum() == len(obs['valid_time']):
    #     raise ValueError('Valid times do not match between SPC Storm Reports and HRRR proxy values!')
    #
    # WRF_proxy_ds = load_WRF_proxy_data(beg=start_date,
    #                                    end=end_date,
    #                                    variables=WRF_proxy_vars,
    #                                    max_forecast_len=max_forecast_len,
    #                                    WRF_data_path=WRF_data_path)
    #
    # if not (obs['valid_time'].values == WRF_proxy_ds['valid_time'].values).sum() == len(obs['valid_time']):
    #     raise ValueError('Valid times do not match between SPC Storm Reports and WRF proxy values!')
    #
    # HRRR_proxy_quant_df = get_quantiles(data=HRRR_proxy_ds,
    #                                     quantiles=proxy_quantile_vals)
    #
    # WRF_proxy_quant_df = get_quantiles(data=WRF_proxy_ds,
    #                                    quantiles=proxy_quantile_vals)
    #
    # HRRR_proxy_quant_df.to_csv(join(dated_output_path, 'HRRR_proxy_quantile_vals.csv'))
    # WRF_proxy_quant_df.to_csv(join(dated_output_path, 'WRF_proxy_quantile_vals.csv'))
    #
    # HRRR_proxy_events = get_proxy_events(data=HRRR_proxy_ds,
    #                                      quantile_df=HRRR_proxy_quant_df,
    #                                      variables=HRRR_proxy_vars,
    #                                      proj_str=config["proj_str"]
    #                                      model_grid_path=model_grid_path,
    #                                      use_saved_indices=use_saved_indices,
    #                                      index_path=HRRR_indices_path)
    #
    # if not (obs['valid_time'].values == HRRR_proxy_events['valid_time'].values).sum() == len(obs['valid_time']):
    #     raise ValueError('Valid times do not match between SPC Storm Reports and proxy events!')
    # HRRR_proxy_events.to_netcdf(join(dated_output_path, f'HRRR_proxy_events.nc'))
    #
    # WRF_proxy_events = get_proxy_events(data=WRF_proxy_ds,
    #                                     quantile_df=WRF_proxy_quant_df,
    #                                     variables=WRF_proxy_vars,
    #                                     model_grid_path=model_grid_path,
    #                                     use_saved_indices=True,
    #                                     index_path=WRF_indices_path)
    #
    # if not (obs['valid_time'].values == WRF_proxy_events['valid_time'].values).sum() == len(obs['valid_time']):
    #     raise ValueError('Valid times do not match between SPC Storm Reports and proxy events!')
    # WRF_proxy_events.to_netcdf(join(dated_output_path, f'WRF_proxy_events.nc'))
    print(f'Post processing completed in {(time.time() - start_time)/60:0.1f} minutes.')

if __name__ == "__main__":
    main()