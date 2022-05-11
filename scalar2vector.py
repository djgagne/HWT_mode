import logging
import numpy as np
import pandas as pd
import sys

def scalar2vector(*arg):
    logging.error("scalar2vector.scalar2vector() is depreciated. Use scalar2vector.uvmagnitude() instead.")
    sys.exit(1)

# Used to be named scalar2vector
# Given U and V components, return magnitude.
def uvmagnitude(df, drop=True):
    possible_features = [f'SHR{z}{potential}{sfx}' for z in "136" for potential in ["","-potential"] for sfx in ["_min","_mean","_max"]]
    possible_features += [f'10{potential}{sfx}' for potential in ["","-potential"] for sfx in ["_min","_mean","_max"]]
    logging.debug(f"uvmagnitude: possible_features={possible_features}")
    # Find these u/v features
    for possible_feature in possible_features:
        uc = "U"+possible_feature 
        vc = "V"+possible_feature 
        if uc in df.columns and vc in df.columns:
            logging.info(f"calculate {possible_feature} magnitude")
            df[possible_feature] = ( df[uc]**2 + df[vc]**2 )**(0.5)
            if drop:
                logging.debug(f"drop {[uc,vc]} columns from dataframe")
                df = df.drop(columns=[uc,vc])
    return df

