import numpy as np
from scipy import stats
rng = np.random.default_rng()

rvs1 = stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
rvs2 = (stats.norm.rvs(loc=5, scale=10, size=500, random_state=rng)
        + stats.norm.rvs(scale=0.2, size=500, random_state=rng))
a = stats.ttest_rel(rvs1, rvs2)


rvs3 = (stats.norm.rvs(loc=8, scale=10, size=500, random_state=rng)
        + stats.norm.rvs(scale=0.2, size=500, random_state=rng))
stats.ttest_rel(rvs1, rvs3)
