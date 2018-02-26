import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

plantData = pd.read_csv('PlantGrowth.csv')

#show the boxplot for the data:
plantData.boxplot('weight', by='group', figsize=(12, 8))
plt.show()

#for the anova we need to create our one-way model:
#in other words:
#for this data set in the "group" column there are three differences:
#'ctrl,' 'trt1,' and 'trt2' -> 'control,' 'treatment 1,' and 'treatment 2'
#Looking at the boxplot, there appear to be differences in the dried weight for the two treatments.
#So, our model ('weight ~ group') means that we believe that the group affects the weight.
#In other words, we believe (hypothesize) that the weight of the plant is linearly associated with the group
model = ols('weight ~ group', data=plantData).fit()
                 
#actually run the anova:
aov_table = sm.stats.anova_lm(model, typ=2)
print("ANOVA results:")
print(aov_table)

#So, uh, what does the table mean?
#The df (degrees of freedom), the F-value, and the residual are all important to statistians.
#However, if you are completely lost, just look for the p-value.
#In the table we see 'PR(>F) = 0.01591' which means that p < 0.05.
#So, for us, that means that there is a statistical difference between the two treatments of ferterlizer  and the control

#Now that we know that there is a difference, what is there a difference between?
#So, we will compare the three groups against each other:
print("\nTukey HSD results:")
mc = MultiComparison(plantData['weight'], plantData['group'])
result = mc.tukeyhsd()

print(result)
#As  you can visually see from the boxplot, the results of the Tukey HSD  test show that only the groups 'trt1' and 'trt2' are different from each  other (we can reject the NULL hypothesis).
