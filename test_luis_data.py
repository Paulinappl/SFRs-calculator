import numpy as np
import pandas as pd
import sfr_calculator
import matplotlib.pyplot as plt

data = pd.read_csv("A2670Finalcat.csv")
# print(data.head(4))
mag_data = data[['mag_u', 'mag_g','mag_r','mag_z']]
mag_data.columns = ['u','g','r','z']
mag_data -= 37.68

test_sfrs = sfr_calculator.compute(bands=['u','g','r','z'], user_data=mag_data)
# print(np.round(np.log10(test_sfrs),1))
print(test_sfrs-data['log_SFR'])
