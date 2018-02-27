import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import log

def safe_ln(x):
    if x <=0:
        return 0
    else:
        return log(x)

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

#*****************************************************************************
# Part 2-a  Scatterplot and linear regression for Sex vs Survived
#*****************************************************************************

print("What is the correlation of 'female' to survived? Visualize it with the \
corresponding linear regression. What is the correlation of 'male' to \
survived? Visualize it with the corresponding linear regression.")
print("\tWe will make a scatterplot of Sex vs Survived and calculate the \
regression. This will show us the strength of the relationship between Sex \
and Survived.\n")

# create a dataframe with just Survived and Sex. Use 0=female and 1=male
df_SexSurvived = pd.concat(
        [
            data['Survived'], 
            pd.get_dummies(data['Sex'])['male']
        ], 
        axis=1)
df_SexSurvived = df_SexSurvived.rename(columns={"male":"Sex"})

# calculate r value
rValue = df_SexSurvived['Sex'].corr(df_SexSurvived['Survived'])

# scatterplot
plt.title('Correlation between Sex and Survived')
plt.xlabel('Sex (0=female, 1=male)')
plt.ylabel('Survived (0=false, 1=true)')
plt.text(.5, .8, 'r value: {0:.3f}'.format(rValue))

# regression line
fit = np.polyfit(df_SexSurvived['Sex'], df_SexSurvived['Survived'], 1)
fit_fn = np.poly1d(fit) 

# plot both on a graph
plt.plot(df_SexSurvived['Sex'], df_SexSurvived['Survived'], 'go', 
            df_SexSurvived['Sex'], fit_fn(df_SexSurvived['Sex']), '--k')
plt.show()

# Explanation
print("\tWith an r value of {0:.3f}, we can say that there is a moderate \
relationship between Sex and Survival. Females are more likely to survive \
than males.\n".format(rValue))


###############################################################################
# Part 2-b   Distributions of two columns
###############################################################################

print("Pick two columns for this question: Age and Fare. For the different \
columns, are they all normal distributions? Visualize the distribution. If it \
is not a normal distribution, transform it. Visualize it again.\n")

print("\tAge distribution:")

# Age
plt.title("Age Distribution")
df_Age = data['Age']
df_Age.dropna(inplace=True)
sns.distplot(df_Age)
plt.show()
print("\tThe distribution is pretty normal, with only slight abberations. We \
are OK to proceed.\n")

# Fare
print("\tFare Distribution:")
sns.distplot(data['Fare'])
plt.title("Fare Distribution")
plt.show()
print("\tThe distribution is not normal. We will try taking the natural log \
of Fare.\n")

print("\tln(Fare) Distribution:")
data['ln_Fare'] = data['Fare'].map(safe_ln)
sns.distplot(data['ln_Fare'])
plt.title("Natural Log of Fare Distribution")
plt.xlabel("Natural Log of Fare")
plt.show()
print("\tThe distribution is much more normal. We are OK to proceed.\n")


###############################################################################
# Part 3:  Bivariate Visualizations
###############################################################################

