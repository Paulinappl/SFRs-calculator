#### This is the file with the units and end to end tests ######

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

## only load the bands that correspond to the users bands
def load_tng_data(mag_file, sfr_file, bands_to_use=['u', 'b', 'v', 'k', 'g', 'r', 'i', 'z']): 
    try:
        mags = pd.DataFrame(np.load(mag_file))
    except:
        print(f'Error when loading {mag_file}.')
    try:
        sfrs = pd.DataFrame(np.load(sfr_file))
    except:
        print(f'Error when loading {sfr_file}.')
    mags.columns = ['u', 'b', 'v', 'k', 'g', 'r', 'i', 'z']
    mags_to_use = mags[bands_to_use]
    return mags_to_use, np.clip(np.log10(sfrs),-5,100)

def split_dataset(mags, sfrs):
    mag_train, mag_test, sfr_train, sfr_test = train_test_split(mags, sfrs, test_size=0.15, 
                                                                random_state=12)
    return mag_train, mag_test, sfr_train, sfr_test

# [U B V K g r i z] are the TNG bands
def compute(bands, user_data):
    '''SFR computation

    Calculate the star-formation rates (SFRs) of the user-defined galaxies using the model trained on TNG data.

    Args:
        bands (list): list of strings where each string refers to a different photometric band. These are the photometric bands input by the user.
        user_data (array): absolute magnitudes in each band specified in "bands."

    Returns:
        array: SFRs of the galaxies
    '''
    mags, sfrs = load_tng_data('SubhaloMag.npy','subhaloSFR.npy', bands_to_use=bands)
    mag_train, mag_test, sfr_train, sfr_test = split_dataset(mags, sfrs)
    # print(type(mag_test), len(sfr_train), len(sfr_test), mag_test.shape, mag_train.shape)

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(random_state=0,n_estimators=200).fit(mag_train, np.ravel(sfr_train))
    print(model.score(mag_train, sfr_train))
    print(model.score(mag_test, sfr_test))

    user_sfrs = model.predict(user_data)

    # assert user_sfrs has no nans
    #

    return user_sfrs






