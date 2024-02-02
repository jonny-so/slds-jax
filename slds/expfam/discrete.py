# multi-dimensional generalisation of categorical distribution. note that the operations
# in this file do NOT automatically batch over leading indices as the shape of parameters
# is used to infer the dimensionality of the distribution. 

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np
from jax import Array

DiscreteNatparams = tuple[Array, Array]
DiscreteMeanparams = tuple[Array, Array]

# marginalise out masked dimensions
def discrete_marginalise(natparams, mask):
    assert(mask.ndim == 1)
    assert(mask.shape[0] == natparams.ndim)
    alpha = jax.scipy.special.logsumexp(natparams, axis=np.arange(len(mask))[mask])
    _logZ = discrete_logZ(alpha)
    return alpha - _logZ, _logZ

# condition on masked dimensions
def discrete_condition(natparams, mask, x):
    assert(x.ndim == 1)
    assert(x.shape[0] < natparams.ndim)
    axes = np.concatenate([np.arange(len(mask))[mask], np.arange(len(mask))[~mask]])
    res = jnp.transpose(natparams, axes)[tuple(x) if x.ndim > 0 else x]
    return res

def discrete_normalise(natparams):
    return natparams - jax.scipy.special.logsumexp(natparams)

def discrete_natparams(meanparams):
    return jnp.log(meanparams)

def discrete_meanparams(natparams):
    return jnp.exp(discrete_normalise(natparams))

def discrete_logZ(natparams):
    return jax.scipy.special.logsumexp(natparams)

def discrete_dot(natparams, stats):
    assert(natparams.shape == stats.shape)
    return jnp.sum(natparams*stats)

def discrete_stats(x, shape):
    return jnp.zeros(shape).at[tuple(x)].set(1.0)

def discrete_logprob(natparams, x):
    return discrete_normalise(natparams)[tuple(x)]

def discrete_sample(rng, natparams, shape_prefix=()):
    x = jax.random.categorical(rng, natparams.reshape(-1), shape=shape_prefix)
    res = jnp.stack(jnp.unravel_index(x, natparams.shape), -1)
    return res

def discrete_entropy(natparams):
    p = discrete_meanparams(natparams)
    return -jnp.sum(p*jnp.log(p))