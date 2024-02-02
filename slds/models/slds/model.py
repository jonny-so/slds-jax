import jax
import jax.numpy as jnp
from jax import Array, vmap
from jax.lax import scan
from jax.random import split
from jax.typing import ArrayLike
from typing import NamedTuple

# generative model description
class Slds(NamedTuple):
    A: Array # discrete state transition probabilities [K,K]
    a0: Array # initial discrete state probabilities [K]
    B: Array # continuous state transition matrices [K,Dz,Dz]
    b: Array # continuous state transition biases [K,Dz]
    b0: Array # initial continuous state means [K,Dz]]
    C: Array # emission matrices [K,Dz,Dx]
    c: Array # emission biases [K,Dx]
    Q: Array # transition precisions [K,Dz,Dz]
    Q0: Array # initial continuous state precisions [K,Dz,Dz]
    R: Array # emission precisions [K,Dx,Dx]

# sample from generative model
def slds_sample(key: Array, model: Slds, T: int):

    A, a0, B, b, b0, C, c, Q, Q0, R = model

    def _sample_s_t(logits, k):
        s = jax.random.categorical(k, logits)
        return jnp.log(A[s]), s
    def _sample_z_t(mV, _):
        k, s = _
        z = jax.random.multivariate_normal(k, mV[0][s], mV[1][s])
        msg = B.dot(z) + b, jnp.linalg.inv(Q)
        return msg, z
    def _sample_x_t(k, s, z):
        return jax.random.multivariate_normal(k, C[s].dot(z) + c[s], jnp.linalg.inv(R[s]))

    key_s, key_z, key_x = split(key, 3)
    s = scan(_sample_s_t, jnp.log(a0), jax.random.split(key_s, T))[1]
    z = scan(_sample_z_t, (b0, jnp.linalg.inv(Q0)), (jax.random.split(key_z, T), s))[1]
    x = vmap(_sample_x_t)(jax.random.split(key_x, T), s, z)

    return s, z, x