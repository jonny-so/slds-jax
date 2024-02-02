import scipy.stats

from slds.expfam.gmm import *

from slds.util import *
from jax import grad, vmap
from jax.random import split

jax.config.update("jax_enable_x64", True)

class TestGmm():

    rng = jax.random.PRNGKey(0)
    K = 3
    D = 8
    rng, logp = rngcall(rng, jax.random.normal, (K,))
    rng, h = rngcall(rng, jax.random.normal, (K,D,))
    rng, J = rngcall(rng, jax.random.normal, (K, D,D))
    J = -.5*((mmp(transpose(J), J)) + jnp.eye(D)*1e-2)
    logp -= jax.scipy.special.logsumexp(logp)
    gamma = logp - gaussian_logZ((h, J))
    
    natparams = gamma, h, J
    meanparams = gmm_meanparams(natparams)

    def test_natparams_meanparams_consistency(self):
        natparams2 = gmm_natparams(self.meanparams)
        assert jnp.allclose(self.natparams[0], natparams2[0])
        assert jnp.allclose(self.natparams[1], natparams2[1])

    def test_logZ_meanparams_consistency(self):
        # representation is not minimal, so symmetrize to ensure consistency
        meanparams2 = gmm_symmetrize(grad(gmm_logZ)(self.natparams))
        assert jnp.allclose(self.meanparams[0], meanparams2[0])
        assert jnp.allclose(self.meanparams[1], meanparams2[1])

    def test_sample_meanparams_consistency(self):
        nsamples = 10000
        self.rng, x = rngcall(self.rng, gmm_sample, self.natparams, (nsamples,))
        stats = vmap(gmm_stats, (0, None))(x, self.K)
        assert jnp.all(scipy.stats.ttest_1samp((stats[0] - self.meanparams[0]), .0).pvalue > .001)
        assert jnp.all(scipy.stats.ttest_1samp((stats[1] - self.meanparams[1]), .0).pvalue > .001)
        assert jnp.all(scipy.stats.ttest_1samp((stats[2] - self.meanparams[2]), .0).pvalue > .001)

    def test_sample_logprob_entropy_consistency(self):
        nsamples = 10000
        self.rng, x = rngcall(self.rng, gmm_sample, self.natparams, (nsamples,))
        logprob = vmap(partial(gmm_logprob, self.natparams))(x)
        entropy = gmm_entropy(self.natparams)
        assert scipy.stats.ttest_1samp(logprob, -entropy).pvalue > .01
