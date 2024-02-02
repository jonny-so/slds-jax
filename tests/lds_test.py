import scipy.stats

from slds.expfam.lds import *
from slds.expfam.lds import _maskfirst, _masklast
from slds.models.lgssm.arma import ar2_lgssm
from slds.models.lgssm.inference import _joint_natparams
from slds.models.lgssm.model import *

from slds.util import *
from jax import grad

jax.config.update("jax_enable_x64", True)

class TestLds():

    lgssm = ar2_lgssm(alpha=.5, beta=.1, mu=.0, vz=.1, vz0=1., vx=.5**2)
    rng = jax.random.PRNGKey(0)
    rng, (z, x) = rngcall(rng, lgssm_sample, lgssm, T=100)
    natparams = _joint_natparams(lgssm, x)

    def test_meanparams_consistency(self):
        mu_z, mu_zz = lds_meanparams(self.natparams)
        T, D = mu_z[0].shape
        for t in range(T-1):
            # check means
            assert jnp.allclose(mu_z[0][t], mu_zz[0][t, _maskfirst(D, 2*D)])
            assert jnp.allclose(mu_z[0][t+1], mu_zz[0][t, _masklast(D, 2*D)])
            # check covariances
            assert jnp.allclose(mu_z[1][t], submatrix(mu_zz[1][t], _maskfirst(D, 2*D), _maskfirst(D, 2*D)))
            assert jnp.allclose(mu_z[1][t+1], submatrix(mu_zz[1][t], _masklast(D, 2*D), _masklast(D, 2*D)))

    # the LDS representation is not minimal, so we check all duplicated natparams
    def test_logZ_meanparams_consistency(self):
        mu_z, mu_zz = lds_meanparams(self.natparams)
        g_z, g_zz = grad(lds_logZ)(self.natparams)
        T, D = mu_z[0].shape

        # check means
        for t in range(T):
            assert jnp.allclose(mu_z[0][t], g_z[0][t])
            if t > 0:
                assert jnp.allclose(mu_z[0][t], g_zz[0][t-1, _masklast(D, 2*D)])
            if t < T - 1:
                assert jnp.allclose(mu_z[0][t], g_zz[0][t, _maskfirst(D, 2*D)])

        # check covariances
        for t in range(T):
            assert jnp.allclose(mu_z[1][t], symmetrize(g_z[1][t]))
            if t < T - 1:
                assert jnp.allclose(mu_zz[1][t], symmetrize(g_zz[1][t]))

    def test_sample_meanparams_consistency(self):
        nsamples = 10000
        mu_z, mu_zz = lds_meanparams(self.natparams)
        z = vmap(lds_sample, (0, None))(split(self.rng, nsamples), self.natparams)
        stats = lds_stats(z)
        assert jnp.all(scipy.stats.ttest_1samp((stats[0][0] - mu_z[0]), .0).pvalue > .001)
        assert jnp.all(scipy.stats.ttest_1samp((stats[0][1] - mu_z[1]), .0).pvalue > .001)
        assert jnp.all(scipy.stats.ttest_1samp((stats[1][0] - mu_zz[0]), .0).pvalue > .001)
        assert jnp.all(scipy.stats.ttest_1samp((stats[1][1] - mu_zz[1]), .0).pvalue > .001)

    def test_sample_logprob_entropy_consistency(self):
        nsamples = 10000
        z = vmap(lds_sample, (0, None))(split(self.rng, nsamples), self.natparams)
        logprob = vmap(partial(lds_logprob, self.natparams))(z)
        entropy = lds_entropy(self.natparams)
        assert scipy.stats.ttest_1samp(logprob, -entropy).pvalue > .01

    def test_entropy(self):
        marginals = lds_marginals(self.natparams)
        singleton_entropies = vmap(gaussian_entropy)(marginals[0])
        pairwise_entropies = vmap(gaussian_entropy)(marginals[1])
        entropy = lds_entropy(self.natparams)
        assert jnp.isclose(entropy, pairwise_entropies.sum() - singleton_entropies[1:-1].sum())