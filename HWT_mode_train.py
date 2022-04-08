#!/usr/bin/env python

def baseline_model(input_dim=None, name=None,numclasses=None, neurons=16, layer=2, optimizer='adam', dropout=0):

    # Discard any pre-existing version of the model.
    model = Sequential(name=name)
    model.add(Dense(neurons, input_dim=input_dim, activation='relu', name="storm_and_env_features"))
    for i in range(layer-1):
        model.add(Dropout(rate=dropout))
        model.add(Dense(neurons, activation='relu'))
    model.add(Dropout(rate=dropout))
    if numclasses > 1:
        model.add(Dense(numclasses, activation='softmax', name="predictions")) # class probabilities add to 1
    else:
        model.add(Dense(numclasses, activation='sigmoid')) 

    # Compile model with optimizer and loss function. MSE is same as brier_score.
    loss="binary_crossentropy"
    if numclasses > 1:
        loss="categorical_crossentropy"
    model.compile(loss=loss, optimizer=optimizer, metrics=[MeanSquaredError(), brier_skill_score, AUC(), "accuracy"])

    return model

def get_other_validation_data(debug=False):
    searchpath = "/glade/p/cisl/aiml/hwt_mode/HWT_2021/WRF_Objects/track_data_wrf_3km_csv_refl/track_step*.csv" # missing many features
    searchpath = "/glade/scratch/ahijevyc/track_data_ncarstorm_3km_REFL_COM_hyst_csv/patch_radius_48/track_step*_d01_2021*.csv"
    searchpath = "/glade/scratch/ahijevyc/track_data_ncarstorm_3km_REFL_COM_hyst_csv/track_step*_d01_20*.csv"
    df = pd.concat([pd.read_csv(f, header=0, index_col="Step_ID", parse_dates=["Run_Date","Valid_Date"]) for f in glob.glob(searchpath)])
    df = df.sort_values(["Run_Date", "Forecast_Hour"])
    df = scalar2vector.decompose_circ_feature(df, "orientation", scale2rad=2., debug=debug)
    df.loc[:,"Local_Solar_Hour"] = df["Valid_Hour_UTC"] + df["Centroid_Lon"]/15.
    df = scalar2vector.decompose_circ_feature(df, "Local_Solar_Hour", scale2rad=2.*np.pi/24., debug=debug) # orientation cycles at pi, not 2*pi
    df = scalar2vector.uvmagnitude(df, drop=False)
    return df

feature_dict = {
        "long" : ['UP_HELI_MAX_mean',
               'UP_HELI_MAX_max', 'GRPL_MAX_mean', 'GRPL_MAX_max',
               'WSPD10MAX_mean', 'WSPD10MAX_max', 
               'W_UP_MAX_mean', 'W_UP_MAX_max', 'W_DN_MAX_mean',
               'W_DN_MAX_max', 'W_DN_MAX_min', 'RVORT1_MAX_mean', 'RVORT1_MAX_max',
               'RVORT5_MAX_mean', 'RVORT5_MAX_max', 
               'UP_HELI_MAX03_mean', 'UP_HELI_MAX03_max', 
               'UP_HELI_MAX01_mean', 'UP_HELI_MAX01_max', 
               'UP_HELI_MIN_mean', 'UP_HELI_MIN_min', 
               'REFL_COM_mean', 'REFL_COM_max', 'REFL_1KM_AGL_mean',
               'REFL_1KM_AGL_max', 'REFD_MAX_mean', 'REFD_MAX_max',
               'PSFC_mean', 'PSFC_max', 'PSFC_min', 'T2_mean',
               'T2_max', 'T2_min', 'Q2_mean', 'Q2_max', 'Q2_min', 'TD2_mean',
               'TD2_max', 'TD2_min', 'SBLCL-potential_mean', 'SBLCL-potential_max',
               'SBLCL-potential_min', 'MLLCL-potential_mean', 'MLLCL-potential_max',
               'MLLCL-potential_min', 'SBCAPE-potential_mean', 'SBCAPE-potential_max',
               'SBCAPE-potential_min', 'MLCAPE-potential_mean', 'MLCAPE-potential_max',
               'MLCAPE-potential_min', 'MUCAPE-potential_mean', 'MUCAPE-potential_max',
               'MUCAPE-potential_min', 'SBCINH-potential_mean', 'SBCINH-potential_max',
               'SBCINH-potential_min', 'MLCINH-potential_mean', 'MLCINH-potential_max',
               'MLCINH-potential_min', 'SRH03-potential_mean', 'SRH03-potential_max',
               'SRH03-potential_min', 'SRH01-potential_mean', 'SRH01-potential_max',
               'SRH01-potential_min', 'PSFC-potential_mean', 'PSFC-potential_max',
               'PSFC-potential_min', 'T2-potential_mean', 'T2-potential_max',
               'T2-potential_min', 'Q2-potential_mean', 'Q2-potential_max',
               'Q2-potential_min', 'TD2-potential_mean', 'TD2-potential_max',
               'TD2-potential_min', '10-potential_mean', '10-potential_max',
               '10-potential_min', 'area', 'eccentricity', 'major_axis_length',
               'minor_axis_length', # 'Max_Hail_Size', # Max_Hail_Size always 0.0, hence, normalized value is NaN (divide by zero std)
               "Centroid_Lat", "Centroid_Lon", "Local_Solar_Hour_sin", "Local_Solar_Hour_cos", "orientation_sin", "orientation_cos",
               '10_min', '10_mean', '10_max', 'SHR1-potential_min',
               'SHR1-potential_mean', 'SHR1-potential_max', 'SHR6-potential_min',
               'SHR6-potential_mean', 'SHR6-potential_max'],
        "2021HWT" : ["REFL_COM_mean","REFL_COM_max","REFL_COM_min","UP_HELI_MAX_mean","UP_HELI_MAX_max","UP_HELI_MAX_min",
                "area","eccentricity","major_axis_length","minor_axis_length",
                "Centroid_Lat", "Centroid_Lon", "Local_Solar_Hour_sin", "Local_Solar_Hour_cos", "orientation_sin", "orientation_cos"],
        "basic" : ['UP_HELI_MAX_max', 'major_axis_length', 'UP_HELI_MAX03_max', 'area', 'minor_axis_length',
               'eccentricity', 'UP_HELI_MAX_mean', 'UP_HELI_MAX01_max', 'UP_HELI_MAX01_mean', 'UP_HELI_MAX03_mean', 
               'REFD_MAX_max', 'RVORT1_MAX_max', 'RVORT5_MAX_max', 'REFL_1KM_AGL_max', '10_max', 
               '10-potential_max', 'SHR6-potential_max', 'SHR6-potential_mean', 'WSPD10MAX_max', 'UP_HELI_MIN_mean',
               "Centroid_Lat", "Centroid_Lon"], 
        "basic2" : [ 'UP_HELI_MAX_max', 'major_axis_length', 'UP_HELI_MAX03_max', 'area', 'minor_axis_length',
               'eccentricity', 'UP_HELI_MAX_mean', 'UP_HELI_MAX01_max', 'UP_HELI_MAX01_mean', 'UP_HELI_MAX03_mean', 
               'REFD_MAX_max', 'RVORT1_MAX_max', 'RVORT5_MAX_max', '10_max','SHR1-potential_max','MLLCL-potential_min', 
               '10-potential_max', 'SHR6-potential_max', 'SHR6-potential_mean', 'WSPD10MAX_max', 'UP_HELI_MIN_min',
               "Centroid_Lat", "Centroid_Lon", "Local_Solar_Hour_sin", "Local_Solar_Hour_cos"],
       "basic+circ" : # circular fields like orientation angle and local solar hour
                [ 'UP_HELI_MAX_max', 'major_axis_length', 'UP_HELI_MAX03_max', 'area', 'minor_axis_length',
               'eccentricity', 'UP_HELI_MAX_mean', 'UP_HELI_MAX01_max', 'UP_HELI_MAX01_mean', 'UP_HELI_MAX03_mean', 
               'REFD_MAX_max', 'RVORT1_MAX_max', 'RVORT5_MAX_max', 'REFL_1KM_AGL_max', '10_max', 
               '10-potential_max', 'SHR6-potential_max', 'SHR6-potential_mean', 'WSPD10MAX_max', 'UP_HELI_MIN_mean',
               "Centroid_Lat", "Centroid_Lon", "Local_Solar_Hour_sin", "Local_Solar_Hour_cos", "orientation_sin", "orientation_cos"],
       "long2" : ["Forecast_Hour","Duration","Centroid_Lon","Centroid_Lat","Storm_Motion_U","Storm_Motion_V","UP_HELI_MAX_mean","UP_HELI_MAX_max","UP_HELI_MAX_min","GRPL_MAX_mean","GRPL_MAX_max","GRPL_MAX_min","WSPD10MAX_mean","WSPD10MAX_max","WSPD10MAX_min","W_UP_MAX_mean","W_UP_MAX_max","W_UP_MAX_min","W_DN_MAX_mean","W_DN_MAX_max","W_DN_MAX_min","RVORT1_MAX_mean","RVORT1_MAX_max","RVORT1_MAX_min","RVORT5_MAX_mean","RVORT5_MAX_max","RVORT5_MAX_min","UP_HELI_MAX03_mean","UP_HELI_MAX03_max","UP_HELI_MAX03_min","UP_HELI_MAX01_mean","UP_HELI_MAX01_max","UP_HELI_MAX01_min","UP_HELI_MIN_mean","UP_HELI_MIN_max","UP_HELI_MIN_min","REFL_COM_mean","REFL_COM_max","REFL_1KM_AGL_mean","REFL_1KM_AGL_max","REFL_1KM_AGL_min","REFD_MAX_mean","REFD_MAX_max","REFD_MAX_min","PSFC_mean","PSFC_max","PSFC_min","T2_mean","T2_max","T2_min","Q2_mean","Q2_max","Q2_min","TD2_mean","TD2_max","TD2_min","U10_mean","U10_max","U10_min","V10_mean","V10_max","V10_min","SBLCL-potential_mean","SBLCL-potential_max","SBLCL-potential_min","MLLCL-potential_mean","MLLCL-potential_max","MLLCL-potential_min","SBCAPE-potential_mean","SBCAPE-potential_max","SBCAPE-potential_min","MLCAPE-potential_mean","MLCAPE-potential_max","MLCAPE-potential_min","MUCAPE-potential_mean","MUCAPE-potential_max","MUCAPE-potential_min","SBCINH-potential_mean","SBCINH-potential_max","SBCINH-potential_min","MLCINH-potential_mean","MLCINH-potential_max","MLCINH-potential_min","USHR1-potential_mean","USHR1-potential_max","USHR1-potential_min","VSHR1-potential_mean","VSHR1-potential_max","VSHR1-potential_min","USHR6-potential_mean","USHR6-potential_max","USHR6-potential_min","VSHR6-potential_mean","VSHR6-potential_max","VSHR6-potential_min","U_BUNK-potential_mean","U_BUNK-potential_max","U_BUNK-potential_min","V_BUNK-potential_mean","V_BUNK-potential_max","V_BUNK-potential_min","SRH03-potential_mean","SRH03-potential_max","SRH03-potential_min","SRH01-potential_mean","SRH01-potential_max","SRH01-potential_min","PSFC-potential_mean","PSFC-potential_max","PSFC-potential_min","T2-potential_mean","T2-potential_max","T2-potential_min","Q2-potential_mean","Q2-potential_max","Q2-potential_min","TD2-potential_mean","TD2-potential_max","TD2-potential_min","U10-potential_mean","U10-potential_max","U10-potential_min","V10-potential_mean","V10-potential_max","V10-potential_min","area","eccentricity","major_axis_length","minor_axis_length","SHR1-potential_min","SHR1-potential_mean","SHR1-potential_max","SHR6-potential_min","SHR6-potential_mean","SHR6-potential_max","10_min","10_mean","10_max","10-potential_min","10-potential_mean","10-potential_max"]+
                 ["Local_Solar_Hour_sin", "Local_Solar_Hour_cos", "orientation_sin", "orientation_cos"], # removed REFL_COM_min because it is always 35dBZ, except for small cells, where it is noisy.
       "dist0.5":['Q2_max', 'MUCAPE-potential_max', 'T2_max', 'Centroid_Lat', 'V_BUNK-potential_max', 'WSPD10MAX_max', 'REFL_1KM_AGL_max', 'REFL_COM_max', 'UP_HELI_MAX_max', 'UP_HELI_MAX01_max', 'RVORT1_MAX_mean', 'Forecast_Hour', 'SBCINH-potential_min', 'SHR6-potential_max', 'SRH01-potential_max', 'major_axis_length', 'MUCAPE-potential_min', 'V10_min', 'U10-potential_min', 'VSHR1-potential_min', 'V10-potential_min', 'SHR1-potential_min', 'SHR1-potential_mean', 'SBLCL-potential_min', 'SBLCL-potential_max', 'Centroid_Lon', 'REFD_MAX_mean', 'REFL_1KM_AGL_min', 'REFD_MAX_min', '10_mean', 'U_BUNK-potential_min', 'VSHR6-potential_min', 'V_BUNK-potential_min', 'V10_mean', 'USHR6-potential_max', 'SHR6-potential_min', 'SBCINH-potential_max', 'U10_mean', 'U_BUNK-potential_max', 'eccentricity', 'orientation_sin', 'USHR1-potential_max', 'Local_Solar_Hour_sin', 'Storm_Motion_U', 'Storm_Motion_V', 'Duration', 'orientation_cos', 'UP_HELI_MIN_min', 'W_UP_MAX_mean', 'GRPL_MAX_mean', 'W_DN_MAX_max', 'GRPL_MAX_min', 'UP_HELI_MAX03_min', 'UP_HELI_MAX01_min', 'RVORT1_MAX_min', 'UP_HELI_MIN_max', 'UP_HELI_MAX_min'],
       "dist1"  :['Q2_max', 'Centroid_Lat', 'V_BUNK-potential_max', 'REFL_COM_max', 'UP_HELI_MAX_max', 'RVORT1_MAX_mean', 'Forecast_Hour', 'SBCINH-potential_min', 'major_axis_length', 'U10-potential_min', 'V10-potential_min', 'SHR1-potential_mean', 'SBLCL-potential_max', 'Centroid_Lon', 'REFD_MAX_min', '10_mean', 'U_BUNK-potential_min', 'V10_mean', 'USHR6-potential_max', 'SBCINH-potential_max', 'U10_mean', 'eccentricity', 'orientation_cos', 'W_UP_MAX_mean', 'GRPL_MAX_mean', 'RVORT1_MAX_min'],
       "dist1.5":['Centroid_Lat', 'UP_HELI_MAX_max', 'Forecast_Hour', 'major_axis_length', 'SBLCL-potential_max', 'Centroid_Lon', 'U_BUNK-potential_min', 'eccentricity', 'W_UP_MAX_mean', 'GRPL_MAX_mean'],
       "dist2"  :['Centroid_Lat', 'UP_HELI_MAX_max', 'major_axis_length', 'SBLCL-potential_max', 'eccentricity', 'W_UP_MAX_mean'],
       "dist2.5":['Centroid_Lat', 'W_UP_MAX_mean', 'major_axis_length'],
}


def vx_tble(labels,truth,y_pred): 
    tble = pd.DataFrame(columns=["AUC","BSS"], index=labels.cat.categories)
    for i, label in enumerate(tble.index):
        tble.loc[label,"AUC"] = roc_auc_score(truth[:,i], y_pred[:,i])
        bss =  brier_skill_score(K.constant(truth[:,i]), K.constant(y_pred[:,i]))
        tble.loc[label,"BSS"] = K.get_value(bss)
    tble["AUC"] = tble["AUC"].astype(float) # or it will be object type. which i have trouble averaging together with other dataframes.
    tble["BSS"] = tble["BSS"].astype(float)
    return tble


def label_longname_dict():
    d = {"Q1":"QLCS\nQ1+Q2+S2", "S1":"Supercell\nS1+S3", "D1":"Cell\nD1", "D2":"Cluster\nD2"}
    d = {"Q1":"Bow echo\nQ1", "Q2":"QLCS\nQ2", "S1":"Isolated Supercell\nS1", "S2":"Supercell in line\nS2", "S3":"Supercell in cluster\nS3", "D1":"Cell\nD1", "D2":"Cluster\nD2"}
    d = {"Q1":"QLCS\nQ1+Q2+S2", "S1":"Supercell\nS1+S3", "D1":"Disorganized\nD1+D2"}
    return d

import argparse
import glob
import joblib
import matplotlib.pyplot as plt
from ml_functions import brier_skill_score
import numpy as np
import os
import pandas as pd
import pdb
import pickle
import scalar2vector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold 
from sklearn.preprocessing import StandardScaler, label_binarize
import statisticplot # ahijevyc's module
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import MeanSquaredError, AUC
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import sys, time
import yaml

def main():
    # =============Arguments===================
    parser = argparse.ArgumentParser(description = "train/predict neural network",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', nargs="+", default=[32], type=int, help="nn training batch size")
    parser.add_argument("--clobber", action='store_true', help="overwrite any old outfile, if it exists")
    parser.add_argument("--confmin", type=int, default=1, help="minimum confidence-throw out expert labels in training set with lower confidence")
    parser.add_argument("-d", "--debug", action='store_true')
    parser.add_argument("--dropouts", type=float, nargs="+", default=[0.0], help='fraction of neurons to drop in each hidden layer (0-1)')
    parser.add_argument('--nfits', type=int, default=10, help="number of times to fit (train) model")
    parser.add_argument('--epochs', nargs="+", default=[50], type=int, help="number of training epochs")
    parser.add_argument('--freezetime', type=lambda s: pd.to_datetime(s), default=pd.to_datetime("2021-07-08 18:00"), help="ignore labels created at this time and later")
    parser.add_argument('--labelpick', type=str, choices=["first", "last", "all"], default="all", help="how to handle multiple labels on same storm")
    parser.add_argument('--model_fname', type=str, help="filename of machine learning model")
    parser.add_argument('--neurons', type=int, nargs="+", default=[12], help="number of neurons in each nn layer")
    parser.add_argument('--segmentations', nargs="*", type=str, default=["hyst"], help="storm segmentation approaches")
    parser.add_argument('--splittime', type=lambda s: pd.to_datetime(s), default=pd.to_datetime("2013-06-25"), help="train with storms before this time; test this time and after")
    parser.add_argument('--suite', type=str, default='basic', choices=feature_dict.keys(), help="name for group of features")


    # Assign arguments to simple-named variables
    args = parser.parse_args()
    batch_sizes           = args.batchsize
    clobber               = args.clobber
    confmin               = args.confmin
    debug                 = args.debug
    dropouts              = args.dropouts
    nfit                  = args.nfits
    epochs                = args.epochs
    freezetime            = args.freezetime
    labelpick             = args.labelpick
    savedmodel            = args.model_fname
    neurons               = args.neurons
    segmentations         = args.segmentations
    train_test_split_time = args.splittime
    suite                 = args.suite

    print(args)

    ### NEURAL NETWORK PARAMETERS ###

    dataset = 'NSC'
    trained_models_dir = '/glade/work/ahijevyc/NSC_objects'
    if savedmodel:
        pass
    else:
        savedmodel = f"{suite}." + ".".join(label_longname_dict().keys()) + f".{labelpick}label.confmin{confmin}.{neurons[0]}n.ep{epochs[0]}.bs{batch_sizes[0]}"

    ##################################

    mcd = os.getenv("TMPDIR", "/glade/ahijevyc/scratch/temp") + f"/HWT_mode_output/atts_and_expertlabels_"+".".join(segmentations)+".csv"
    print(f'Reading {mcd}')
    if os.path.exists(mcd):
        df = pd.read_csv(mcd, parse_dates=["Run_Date","Valid_Date","labeltime"])
        df = df.sort_values("labeltime") # so you can pick first (or last) label of each storm
    else:
        print(f"Use ~ahijevyc/bin/hagelslag_obj_pdf.py to make {mcd}")
        sys.exit(1)

    df.info() # tried adding show_counts=True but got TypeError: info() got an unexpected keyword argument 'show_counts'
    features = feature_dict[suite]

    # Confidence filter training set (not testing)
    lowconf = (df.conf < confmin) & (df.Valid_Date < train_test_split_time) # just Training
    #lowconf = df.conf < confmin # Training and testing (not recommended by DJG) 
    print(f"dropping {lowconf.sum()}/{len(df)} expert labels with confidence < {confmin} from training set")
    df = df[~lowconf]

    # Labeltime filter
    nlate = (df.labeltime >= freezetime).sum()
    print(f"dropping {nlate}/{len(df)} labeltimes {freezetime} or later")
    df = df[df.labeltime < freezetime]

    # Multiple labels filter
    if labelpick == "first":
        df = df.groupby("Step_ID").first()
    elif labelpick == "last":
        df = df.groupby("Step_ID").last()
    elif labelpick == "all":
        pass


    # Split labels into training and testing sets (training < train_test_split_time <= testing).
    train_indices = df["Valid_Date"] < train_test_split_time
    test_indices = ~train_indices
    df["split"] = "train"
    df.loc[test_indices,"split"] = "test"
    print(f"{train_indices.sum()} ({100.*train_indices.sum()/len(df):.0f}%) training cases Valid_Date < {train_test_split_time}")
    print(f"{test_indices.sum()} ({100.*test_indices.sum()/len(df):.0f}%) test cases Valid_Date >= {train_test_split_time}")

    labels = df['label'].astype("category")
    labels = labels.cat.reorder_categories(["Q1", "Q2", "S1", "S2", "S3", "D1", "D2"], ordered=True)
    # If you change these, also change label_longname_dict() to have the correct categories.
    if len(label_longname_dict()) == 3:
        labels[labels =="D2"] = "D1" # D1/D2 
        labels[labels =="S3"] = "S1" # S1/S3
        labels[labels =="Q2"] = "Q1" 
        labels[labels =="S2"] = "Q1" # Q1/Q2/S2

    labels = labels.cat.rename_categories(label_longname_dict())

    value_cnts = True
    if value_cnts:
        print("plotting class histogram")
        df["label"] = labels
        bax = df[["label","split"]].value_counts().unstack().plot.bar(grid=True)
        for container in bax.containers:
            bax.bar_label(container)
        plt.tight_layout()
        ofile = ".".join(label_longname_dict().keys())+f".{labelpick}labels.value_counts.png"
        plt.savefig(ofile)
        print(os.path.realpath(ofile))
        plt.clf()

    oneclass = False
    if oneclass:
        labels = labels == oneclass
        numclasses = 1
        onehotlabels = labels.astype(int)


    labels = labels.cat.remove_unused_categories()
    numclasses = labels.nunique()
    print(f"{numclasses} classes")
    # assert all(encoder.classes_ == sorted(labels.cat.categories)) # normalize_and_topn.ipynb assumes labels were encoded alphabetically. TODO: is this still true?
    # Tried LabelEncoder, but it encodes alphabetically and does not honor Pandas category order.
    onehotlabels = label_binarize(labels, classes=labels.cat.categories) # label_binarize over LabelEncoder/LabelBinarizer because it allows you to specify order of classes.

    # Circular features
    df = scalar2vector.decompose_circ_feature(df, "orientation", scale2rad=2., debug=debug) # orientation cycles at pi, not 2*pi
    df.loc[:,"Local_Solar_Hour"] = df["Valid_Hour_UTC"] + df["Centroid_Lon"]/15.
    df = scalar2vector.decompose_circ_feature(df, "Local_Solar_Hour", scale2rad=2.*np.pi/24., debug=debug) # orientation cycles at pi, not 2*pi
    for f in ["orientation", "Local_Solar_Hour"]:
        if f in features:
            features += [f+"_sin", f+"_cos"]
            features.remove(f)

    df = df[features]
    
    print("normalize data")
    #scaler = pickle.load(open('/glade/work/ahijevyc/NSC_objects/scaler.pkl', 'rb'))
    #scaler = scaler[features] 
    # We'll want to normalize the whole thing.
    #norm_in_data = (df - scaler.loc["mean"] ) / scaler.loc["std"]
    scaler = StandardScaler().fit(df[train_indices])
    norm_in_data = scaler.transform(df)
    print('done normalizing')

    nn = True
    rf = False
    if nn:
        # train model
        optimizers = ['SGD','rmsprop', 'adam']
        optimizers = ['adam']
        sample_weight = [None]
        class_weight = [None]
        gridsearch = len(optimizers) * len(neurons) * len(epochs) * len(batch_sizes) * len(dropouts) * len(sample_weight) * len(class_weight) > 1 # anything more than 1
        if gridsearch:
            model = KerasClassifier(build_fn=baseline_model, input_dim=norm_in_data.shape[1], numclasses=numclasses, verbose=0)
            param_grid = dict(dropout=dropouts, optimizer=optimizers, num_neurons=neurons, epochs=epochs, batch_size=batch_sizes, sample_weight=sample_weight, class_weight=class_weight)
            grid = GridSearchCV(estimator=model, n_jobs=-1, param_grid = param_grid)

            grid_result = grid.fit(norm_in_data[train_indices], onehotlabels[train_indices], 
                    validation_data=(norm_in_data[test_indices], onehotlabels[test_indices]), verbose=2)
            # summarize results
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
            if clobber or not os.path.exists(savedmodel):
                grid_result.best_estimator_.model.save(savedmodel)
        else:
            models = []
            for i in range(nfit):
                model_i = f"nn/nn_{savedmodel}_{i}"
                if not clobber and os.path.exists(model_i):
                    print(f"loading {model_i}")
                    custom_objects = {"brier_score":brier_score, "brier_skill_score":brier_skill_score}
                    models.append(load_model(model_i, custom_objects=custom_objects))
                    history = models[-1].history
                else:
                    model = baseline_model(input_dim=norm_in_data.shape[1],numclasses=numclasses,neurons=neurons[0], name=f"fit_{i}")
                    history = model.fit(norm_in_data[train_indices], onehotlabels[train_indices], class_weight=class_weight[0], sample_weight=sample_weight[0], 
                        epochs=epochs[0], validation_data=(norm_in_data[test_indices], onehotlabels[test_indices]), batch_size=batch_sizes[0], verbose=2)
                    models.append(model)
                    model.save(model_i)
                    model.save(model_i+".h5", save_format='h5', overwrite=clobber)
                    # Save order of columns 
                    with open(os.path.join(model_i, "columns.yaml"), "w") as file:
                        yaml.dump(
                                dict(columns=scaler.get_feature_names_out().tolist(),
                                    mean=scaler.mean_.tolist(),
                                    std=scaler.scale_.tolist())
                            , file)
                if True and history: # history is None in saved model
                    panel_inches = 4
                    import plot_keras_history
                    fig, ax = plot_keras_history.plot_history(history, path=f"history_{i}.{suite}.png", graphs_per_row=3, side=panel_inches) 
                    fig.clf()

            y_preds = []
            tbles = pd.DataFrame()
            for model in models:
                print(model.name)
                y_pred = model.predict(norm_in_data[test_indices])
                tble = vx_tble(labels, onehotlabels[test_indices], y_pred)
                tbles = tbles.append(tble)
                y_preds.append(y_pred)
            # Kludge to restore order of labels. (QLCS first, not Disorganized)
            tbles = tbles.reset_index()
            tbles["index"] = tbles["index"].astype("category").cat.reorder_categories(labels.cat.categories.values, ordered=True)
            tbles = tbles.groupby("index")
            print("mean of each model fit")
            print(tbles.mean())
            print("std of each model fit")
            print(tbles.std())
            y_pred = np.array(y_preds).mean(axis=0)
            if numclasses == 1:
                y_pred = y_pred[:,0]
            print("skill of the mean model probability")
            print(vx_tble(labels, onehotlabels[test_indices], y_pred))
            statcurves = True
            if statcurves:
                print("reliability diagram, histogram, & ROC curve")
                fig = plt.figure(figsize=(10,7))
                ax1 = plt.subplot2grid((3,2), (0,0), rowspan=2)
                ax2 = plt.subplot2grid((3,2), (2,0), rowspan=1, sharex=ax1)
                ROC_ax = plt.subplot2grid((3,2), (0,1), rowspan=2)
                for i, label in enumerate(labels.cat.categories):
                    reliability_diagram, = statisticplot.reliability_diagram(ax1, labels[test_indices] == label, y_pred[:,i], label=label)
                    counts, bins, patches = statisticplot.count_histogram(ax2, y_pred[:,i])
                    rc = statisticplot.ROC_curve(ROC_ax, labels[test_indices] == label, y_pred[:,i], label=label, fill=False, plabel=False)
                fig.suptitle(f"{suite}")
                fig.text(0.5, 0.01, ' '.join(features), wrap=True, fontsize=6)
                ofile = f"nn/{savedmodel}.statcurves.png"
                fig.savefig(ofile)
                print(os.path.realpath(ofile))
                plt.clf()
            confusion_matrix = True
            if confusion_matrix:
                from scikitplot.metrics import plot_confusion_matrix
                print("confusion matrix")
                fig, ax = plt.subplots()
                y_pred_class = labels.cat.categories[np.argmax(y_pred, axis=1)]
                cm = plot_confusion_matrix(labels[test_indices], y_pred_class, labels.cat.categories, normalize=True, ax=ax)
                fig.suptitle(f"{suite} suite")
                fig.tight_layout()
                ofile = f"nn/{savedmodel}.confusion_matrix.png"
                plt.savefig(ofile)
                print(os.path.realpath(ofile))
                plt.clf()
        df = get_other_validation_data()
        df = df[features]
        norm_in_data2021hwt = scaler.transform(df)
        print('done normalizing another test set')
        preds = []
        print('predicting with multiple models',end='')
        for model in models:
            print('.', end='')
            preds.append(model.predict(norm_in_data2021hwt))
        print()
        mean_pred = np.array(preds).mean(axis=0)
        ofile = f"nn/{savedmodel}.csv"
        df.join(pd.DataFrame(mean_pred, index=df.index, columns=labels.cat.categories)).to_csv(ofile)
        print("created",os.path.realpath(ofile))

    if rf:
        rf_savedmodel = "rf/rf_"+savedmodel
        print(rf_savedmodel)
        if not clobber and os.path.exists(rf_savedmodel):
            print("loading saved rf")
            rf = joblib.load(rf_savedmodel)
        else:
            rf = RandomForestClassifier(n_estimators=200, max_depth=50, min_samples_split=2, oob_score=True, criterion="gini", random_state=None, n_jobs=-1)
            rf.fit(norm_in_data[train_indices], onehotlabels[train_indices])
        if clobber or not os.path.exists(rf_savedmodel):
            joblib.dump(rf, rf_savedmodel)
        #scores = cross_val_score(rf, norm_in_data, onehotlabels, cv=5)
        y_pred = rf.predict(norm_in_data[test_indices])
        print(vx_tble(labels, onehotlabels[test_indices], y_pred))
        importances = pd.Series(rf.feature_importances_ , index=features).sort_values(ascending=False).reset_index()
        print(importances.head(80))
        df = get_other_validation_data()
        norm_in_data = scaler.transform(df)
        print('done normalizing')
        preds = rf.predict(norm_in_data)
        df.join(pd.DataFrame(preds, index=df.index, columns=labels.cat.categories)).to_csv(f"{rf_savedmodel}.csv")
if __name__ == "__main__":
    main()
