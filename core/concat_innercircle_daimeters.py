import glob
import ipdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
files = glob.glob(r"C:\Users\listo\Downloads\innercircle_diameter_results\*.csv")


for i,file in enumerate(files):
    if i==0:
        df=pd.read_csv(file)
    else:
        dfoh=pd.read_csv(file)
        df=pd.concat((df,dfoh))
    
df = df.sort_values(by='percent')
plt.figure(figsize=(10,10))
plt.plot(np.asarray(df["percent"]),np.asarray(df["percent_time_inner_average"]))
plt.xlabel('The Size of inner circle where 0 is smallest')
plt.ylabel('Percent time inside of inner circle')
plt.show()
ipdb.set_trace()
