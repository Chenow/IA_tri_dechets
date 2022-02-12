import scipy
from params.py import *

def test_size(a_m, a_c, alpha):
    s=a_m(1-a_m)
    q=scipy.stats.norm.ppf(alpha, a_m, s)
    margin=(a_m-a_c)^2
    return (q^2)*s/(margin^2)