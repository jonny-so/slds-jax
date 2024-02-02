# linear gaussian state-space model: lds + gaussian observations

import jax
import jax.numpy as jnp
from jax import Array
from jax.lax import scan
from jax.random import split
from jax.typing import ArrayLike
from typing import NamedTuple

# generative model description
class Lgssm(NamedTuple):
    B: Array # transition matrix
    b: Array # transition bias
    b0: Array # initial state mean
    C: Array # emission matrix
    c: Array # emission bias
    Q: Array # transition precision
    Q0: Array # initial state precision
    R: Array # emission precision

# sample from generative model
def lgssm_sample(key: ArrayLike, model: Lgssm, T: int):
    B, b, b0, C, c, Q, Q0, R = model
    def _step(mV, k):
        key_z, key_x = split(k)
        z = jax.random.multivariate_normal(key_z, *mV)
        x = jax.random.multivariate_normal(key_x, C.dot(z) + c, jnp.linalg.inv(R))
        return (B.dot(z) + b, jnp.linalg.inv(Q)), (z, x)
    z, x = scan(_step, (b0, jnp.linalg.inv(Q0)), split(key, T))[1]
    return z, x