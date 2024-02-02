# hidden markov model: dds + categorical observations

import jax
import jax.numpy as jnp
from jax import Array
from jax.lax import scan
from jax.random import split
from jax.typing import ArrayLike
from typing import NamedTuple
from slds.util import rngcall

# generative model description
class Hmm(NamedTuple):
    A: Array # transition probabilities
    a0: Array # initial state probabilities
    C: Array # emission probabilities

# sample from generative model
def hmm_sample(key: ArrayLike, model: Hmm, T: int):
    A, a0, C = model
    def _step(logits, k):
        key_s, key_x = split(k)
        s = jax.random.categorical(key_s, logits=logits)
        x = jax.random.categorical(key_x, logits=jnp.log(C[s]))
        return jnp.log(A[s]), (s, x)
    s, x = scan(_step, jnp.log(a0), jax.random.split(key, T))[1]
    return s, x