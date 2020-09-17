from evaluation_utils.modeling import get_ks_stat, get_wasserstein_distance_pdf
import numpy as np


def test_scale_invariance():

    def check(f):
        m = f(orig, gen)
        m_l = f(orig * 100, gen * 100)
        m_a = f(orig + 10000, gen + 10000)
        assert np.isclose(m, m_l)
        assert np.isclose(m, m_a, atol=1e-2)

    orig = np.random.random(1000)
    gen = np.random.normal(size=1000) - .1
    check(get_ks_stat)
    check(get_wasserstein_distance_pdf)
