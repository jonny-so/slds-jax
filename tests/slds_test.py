from slds.expfam.lds import *
from slds.models.lgssm.arma import ar2_lgssm
from slds.models.slds.model import *
from slds.models.slds.inference_smf import *

from slds.util import *

jax.config.update("jax_enable_x64", True)

class TestSlds():

    rng = jax.random.PRNGKey(0)

    A = jnp.array([
        [.98, .01, .01],
        [.01, .98, .01],
        [.01, .01, .98]])
    a0 = jnp.ones(3) / 3

    B, b, b0, C, c, Q, Q0, R = tree_stack([
        ar2_lgssm(alpha=.9, beta=.1, mu=-1., vz=.001, vz0=1., vx=.5**2),
        ar2_lgssm(alpha=.0, beta=1.0, mu=0., vz=.05, vz0=1., vx=.5**2),
        ar2_lgssm(alpha=.6, beta=.2, mu=1., vz=.01, vz0=1., vx=.5**2)], 0)

    model = Slds(A, a0, B, b, b0, C, c, Q, Q0, R)

    rng, (s, z, x) = rngcall(rng, slds_sample, model, T=100)
    inference_state = slds_smf_inference_init(model, x)

    def test_elbo_increasing(self):
        def _step(s, _):
            return slds_smf_inference_update(s), slds_smf_elbo(s)
        elbos = scan(_step, self.inference_state, jnp.arange(100))[1]
        assert jnp.all(elbos[:-1] - elbos[1:] < 1e-8)

    def test_elbo_convergence(self):
        def _step(s, _):
            return slds_smf_inference_update(s), slds_smf_elbo(s)
        elbos = scan(_step, self.inference_state, jnp.arange(100))[1]
        assert jnp.abs(elbos[-1] - elbos[-2]) < 1e-8