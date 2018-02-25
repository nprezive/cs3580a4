import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# import data file
data = pd.read_csv("train.csv")

#**********************************
#Part 1-a
#**********************************
print("Perform an ANOVA on the 'Sex' column using 'Survived' as the \
independent variable. What did you find? What does it mean?\n")
print("\tWe will work with the null hypothesis: There is no relationship \
between the groups of the Sex column and passenger survival.\n")

# create a dataframe for each sex containing the survived column
sexes = pd.unique(data.Sex.values)
d_data = {grp:data['Survived'][data.Sex == grp] for grp in sexes}

# perform an ANOVA on Sex using Survived as the dependent variable
F, p = stats.f_oneway(d_data['male'], d_data['female'])

# Explanation
print("\tResults:\n\tF value: {0:.3f}\n\tp value: {1:.3e}\n".format(F,p))
print("\tThe p value is less than .05, so we can reject the null hypothesis \
and safely say there IS a relationship between the groups of the Sex column \
and passenger survival.\n")

#**********************************
#Part 1-b
#**********************************
print("Perform a similiar ANOVA on PClass using 'Survived' as the independent \
variable. What did you find? What does it mean?\n")
print("\tWe will work with the null hypothesis: There is no relationship \
between the groups of the Pclass column and passenger survival.\n")

# create a dataframe for each passenger class containing the survived column
pclasses = pd.unique(data.Pclass.values)
d_data2 = {grp:data['Survived'][data.Pclass == grp] for grp in pclasses}

# Perform an ANOVA on Pclass using Survived as the dependent variable
F, p = stats.f_oneway(d_data2[1], d_data2[2], d_data2[3])

# Explanation
print("\tResults:\n\tF value: {0:.3f}\n\tp value: {1:.3e}\n".format(F,p))
print("\tThe p value is less than .05, so we can reject the null hypothesis \
and safely say there IS a relationship between the groups of the Pclass \
column and passenger survival.\n")

#*********************************
# Part 2-a
#*********************************
