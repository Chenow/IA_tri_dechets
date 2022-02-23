from scipy.stats import norm
from params import *


def test_size(a_m, a_c, alpha):
    s = a_m*(1-a_m)
    q = norm.ppf(alpha/2)
    margin = (a_m - a_c)
    return (q**2)*s/(margin**2) 


def interval_confiance(accuracy, nbr_test, alpha):
    variance = (accuracy*(1 - accuracy))**(1/2)
    q = abs(norm.ppf(alpha/2))
    a = q*variance/nbr_test**(1/2)
    print(f"interval de confiance à {100*(1 - alpha)}% : [{accuracy - a} ; {accuracy + a}]")