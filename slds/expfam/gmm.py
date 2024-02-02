# gaussian mixture model:
#   s ~ discrete
#   x ~ N(\mu_s, V_s)
# note the *joint* distribution over s and x is an exponential family.

import jax
import jax.numpy as jnp
from jax.random import split
from slds.expfam.discrete import discrete_entropy
from slds.expfam.gaussian import *
from slds.util import mvp, transpose

GmmNatparams = tuple[Array, Array, Array]
GmmMeanparams = tuple[Array, Array, Array]

def gmm_meanparams(natparams):
    gamma, h, J = natparams
    J = .5*(J + transpose(J))
    logZ = gaussian_logZ((h, J))
    p = jnp.exp(gamma + logZ - jax.scipy.special.logsumexp(gamma + logZ))
    m = gaussian_meanparams((h, J))
    return p, p[...,None]*m[0], p[...,None,None]*m[1]

def gmm_natparams(meanparams):
    p, px, pxx = meanparams
    x, xx = px/p[...,None], pxx/p[...,None,None]
    h, J = gaussian_natparams((x, xx))
    gamma = jnp.log(p) - gaussian_logZ((h, J))
    return gamma, h, J

def gmm_stats(_, K):
    s, x = _
    ds = jax.nn.one_hot(s, num_classes=K)
    x, xx = gaussian_stats(x)
    dsx = ds[...,None]*x
    dsxx = ds[...,None,None]*xx
    return ds, dsx, dsxx

def gmm_sample(key, natparams, shape=()):
    expand = lambda _: jnp.tile(_, shape + (1,)*_.ndim)
    natparams = tree_map(expand, natparams)
    gamma, h, J = natparams
    logp = gamma + gaussian_logZ((h, J))
    K = gamma.shape[-1]
    key_s, key_x = split(key)
    s = jax.random.categorical(key_s, logp)
    h = jnp.sum(h*jax.nn.one_hot(s, K)[...,None], -2)
    J = jnp.sum(J*jax.nn.one_hot(s, K)[...,None,None], -3)
    V = jnp.linalg.inv(-2*J)
    mu = mvp(V, h)
    x = jax.random.multivariate_normal(key_x, mu, V)
    return s, x

def gmm_standardparams(natparams):
    gamma, h, J = natparams
    logZ = gaussian_logZ((h, J))
    p = jnp.exp(gamma + logZ - jax.scipy.special.logsumexp(gamma + logZ))
    m = gaussian_standardparams((h, J))
    return p, m[0], m[1]

def gmm_logZ(natparams):
    gamma, h, J = natparams
    logZ = gaussian_logZ((h, J))
    assert(gamma.shape == logZ.shape)
    return jax.scipy.special.logsumexp(gamma + logZ)

def gmm_dot(natparams, stats):
    return (
        jnp.sum(natparams[0]*stats[0], -1) + 
        jnp.sum(natparams[1]*stats[1], (-1,-2)) + 
        jnp.sum(natparams[2]*stats[2], (-1,-2,-3)))

def gmm_logprob(natparams, x):
    K = natparams[0].shape[-1]
    return gmm_dot(natparams, gmm_stats(x, K)) - gmm_logZ(natparams)

def gmm_entropy(natparams):
    h, J = natparams[1:]
    p = gmm_meanparams(natparams)[0]
    entropy = jnp.sum(-p*jnp.log(p), -1) + jnp.sum(p*gaussian_entropy((h, J)), -1)
    return entropy

def gmm_symmetrize(natparams):
    gamma, h, J = natparams
    J = .5*(transpose(J) + J)
    return gamma, h, J
