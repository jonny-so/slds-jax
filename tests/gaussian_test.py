import scipy.stats
import numpy.random as npr

from slds.expfam.gaussian import *

from slds.util import *
from jax import grad, vmap
from jax.random import split

jax.config.update("jax_enable_x64", True)

class TestGaussian():

    rng = jax.random.PRNGKey(0)
    D = 8
    rng, h = rngcall(rng, jax.random.normal, (D,))
    rng, J = rngcall(rng, jax.random.normal, (D,D))
    J = -.5*((J.T @ J) + jnp.eye(D)*1e-2)
    
    natparams = h, J
    meanparams = gaussian_meanparams(natparams)

    def test_natparams_meanparams_consistency(self):
        natparams2 = gaussian_natparams(self.meanparams)
        assert jnp.allclose(self.natparams[0], natparams2[0])
        assert jnp.allclose(self.natparams[1], natparams2[1])

    def test_logZ_meanparams_consistency(self):
        # representation is not minimal, so symmetrize to ensure consistency
        meanparams2 = gaussian_symmetrize(grad(gaussian_logZ)(self.natparams))
        assert jnp.allclose(self.meanparams[0], meanparams2[0])
        assert jnp.allclose(self.meanparams[1], meanparams2[1])

    def test_sample_meanparams_consistency(self):
        nsamples = 10000
        self.rng, x = rngcall(self.rng, gaussian_sample, self.natparams, (nsamples,))
        stats = vmap(gaussian_stats)(x)
        assert jnp.all(scipy.stats.ttest_1samp((stats[0] - self.meanparams[0]), .0).pvalue > .001)
        assert jnp.all(scipy.stats.ttest_1samp((stats[1] - self.meanparams[1]), .0).pvalue > .001)

    def test_sample_logprob_entropy_consistency(self):
        nsamples = 10000
        self.rng, x = rngcall(self.rng, gaussian_sample, self.natparams, (nsamples,))
        logprob = vmap(partial(gaussian_logprob, self.natparams))(x)
        entropy = gaussian_entropy(self.natparams)
        assert scipy.stats.ttest_1samp(logprob, -entropy).pvalue > .01

    def test_marginalise(self):
        mask = npr.permutation(np.array([True]*3 + [False]*(self.D-3)))
        marginal_natparams = gaussian_marginalise(self.natparams, mask)[0]
        marginal_meanparams2 = self.meanparams[0][~mask], submatrix(self.meanparams[1], ~mask, ~mask)
        marginal_natparams2 = gaussian_natparams(marginal_meanparams2)
        assert jnp.allclose(marginal_natparams[0], marginal_natparams2[0])
        assert jnp.allclose(marginal_natparams[1], marginal_natparams2[1])
