import pytest
import scipy.stats

from slds.expfam.discrete import *

from slds.util import *
from jax import grad, vmap
from jax.random import split
from itertools import product

jax.config.update("jax_enable_x64", True)

class TestDiscrete():

    rng = jax.random.PRNGKey(0)
    D1 = 8
    D2 = 4
    rng, natparams = rngcall(rng, jax.random.normal, (D1,D2))
    meanparams = discrete_meanparams(natparams)

    def test_meanparams(self):
        p = jnp.exp(self.natparams) / jnp.sum(jnp.exp(self.natparams))
        assert jnp.allclose(self.meanparams, p)

    def test_natparams_meanparams_consistency(self):
        natparams2 = discrete_natparams(self.meanparams)
        assert jnp.allclose(
            discrete_normalise(self.natparams),
            discrete_normalise(natparams2))

    def test_logZ_meanparams_consistency(self):
        meanparams2 = grad(discrete_logZ)(self.natparams)
        assert jnp.allclose(self.meanparams, meanparams2)

    def test_sample_meanparams_consistency(self):
        nsamples = 10000
        self.rng, x = rngcall(self.rng, discrete_sample, self.natparams, (nsamples,))
        stats = vmap(discrete_stats, (0, None))(x, (self.D1, self.D2))
        assert jnp.all(scipy.stats.ttest_1samp((stats - self.meanparams), .0).pvalue > .001)

    def test_sample_logprob_entropy_consistency(self):
        nsamples = 10000
        self.rng, x = rngcall(self.rng, discrete_sample, self.natparams, (nsamples,))
        logprob = vmap(partial(discrete_logprob, self.natparams))(x)
        entropy = discrete_entropy(self.natparams)
        assert scipy.stats.ttest_1samp(logprob, -entropy).pvalue > .01

    def test_marginalise(self):
        p = jnp.exp(self.natparams) / jnp.sum(jnp.exp(self.natparams))
        p1 = p.sum(1)
        p2 = p.sum(0)
        marginal1 = discrete_marginalise(self.natparams, np.array([False, True]))[0]
        marginal2 = discrete_marginalise(self.natparams, np.array([True, False]))[0]
        assert jnp.allclose(discrete_meanparams(marginal1), p1)
        assert jnp.allclose(discrete_meanparams(marginal2), p2)

    def test_condition(self):
        p = jnp.exp(self.natparams) / jnp.sum(jnp.exp(self.natparams))
        for i, j in product(range(self.D1), range(self.D2)):
            p1 = p[:,i] / jnp.sum(p[:,i])
            p2 = p[j] / jnp.sum(p[j])
            conditional1 = discrete_condition(self.natparams, np.array([False, True]), jnp.array([i]))
            conditional2 = discrete_condition(self.natparams, np.array([True, False]), jnp.array([j]))
            assert jnp.allclose(discrete_meanparams(conditional1), p1)
            assert jnp.allclose(discrete_meanparams(conditional2), p2)