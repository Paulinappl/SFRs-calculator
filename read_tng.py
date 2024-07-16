import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_tng_data(mag_file, sfr_file): 
    try:
        mags = pd.DataFrame(np.load(mag_file))
    except:
        print(f'Error when loading {mag_file}.')
    try:
        sfrs = pd.DataFrame(np.load(sfr_file))
    except:
        print(f'Error when loading {sfr_file}.')
    return mags, sfrs


def split_dataset(mags, sfrs):
    mag_train, mag_test, sfr_train, sfr_test = train_test_split(mags, sfrs, test_size=0.15, 
                                                                random_state=12)
    
    return mag_train, mag_test, sfr_train, sfr_test
    




if __name__ == '__main__':
    mags, sfrs = load_tng_data('/Users/smckay/Downloads/tngdata/SubhaloMag.npy','/Users/smckay/Downloads/tngdata/subhaloSFR.npy')
    mag_train, mag_test, sfr_train, sfr_test = split_dataset(mags, sfrs)
    print(type(mag_test), len(sfr_train), len(sfr_test), mag_test.shape, mag_train.shape)

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(random_state=0,n_estimators=200).fit(mag_train, np.ravel(sfr_train))
    print(model.score(mag_train, sfr_train))
    print(model.score(mag_test,sfr_test))
    print(type(np.ravel(sfr_train)))