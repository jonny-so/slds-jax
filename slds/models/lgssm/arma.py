import jax.numpy as jnp
from slds.models.lgssm.model import Lgssm

# alpha: momentum
# beta: mean reversion strength
# mu: mean level
# vz: innovation variance
# vz0: initial state variance
# vx: emission variance
def ar2_lgssm(alpha, beta, mu, vz, vz0, vx):
    B = jnp.array([[1 + alpha - beta, -alpha], [1.0, .0]])
    b = jnp.array([beta * mu, .0])
    b0 = jnp.array([mu, mu])
    C = jnp.array([[1., .0]])
    c = jnp.array([0.])
    Q = jnp.diag(1 / jnp.array([vz, vz*1e-2]))
    Q0 = jnp.diag(1 / jnp.array([vz0, vz0]))
    R = jnp.diag(1 / jnp.array([vx]))
    return Lgssm(B, b, b0, C, c, Q, Q0, R)
