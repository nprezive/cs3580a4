import pandas as pd
from scipy import stats

data = pd.read_csv("train.csv")
sexes = pd.unique(data.Sex.values)
d_data = {grp:data['Survived'][data.Sex == grp] for grp in sexes}
F, p = stats.f_oneway(d_data['male'], d_data['female'])
print(F, p)

pclasses = pd.unique(data.Pclass.values)
print(pclasses)
d_data2 = {grp:data['Survived'][data.Pclass == grp] for grp in pclasses}
F, p = stats.f_oneway(d_data2[1], d_data2[2], d_data2[3])
print(F, p)