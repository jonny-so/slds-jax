# discrete dynamical system:
#   p(s_0 = i) = [a_0]_i
#   p(s_t = j | s_{t-1} = i) = A_{ij}

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import random
import numpy
from typing import NamedTuple, Union
from jax import Array, vmap
from jax.lax import scan
from jax.random import split

from slds.expfam.discrete import *
from slds.util import *

DdsNatparams = tuple[DiscreteNatparams, DiscreteNatparams]
DdsMeanparams = tuple[DiscreteMeanparams, DiscreteMeanparams]

class DdsMarginals(NamedTuple):
    singletons: DiscreteNatparams
    pairwise: DiscreteNatparams

_maskfirst = lambda: np.array([True, False])
_masklast = lambda: np.array([False, True])

def _join_natparams(eta_a, eta_b, eta_ab):
    return eta_a[:,None] + eta_b[None] + eta_ab

def _forward_message(mfwd, eta_transition):
    m, logZ = discrete_marginalise(mfwd[0][:,None] + eta_transition, _maskfirst())
    return m, logZ + mfwd[1]

def _backward_message(mbwd, eta_transition):
    m, logZ = discrete_marginalise(mbwd[0][None] + eta_transition, _masklast())
    return m, logZ + mbwd[1]

def _forward_step(m_in, eta_t):
    eta_in, scale_in = m_in
    eta_s_t, eta_ss_t = eta_t
    m_out = _forward_message((tree_add(eta_in, eta_s_t), scale_in), eta_ss_t)
    return m_out, m_out
    
def _backward_step(m_in, eta_t):
    eta_in, scale_in = m_in
    eta_s_t, eta_ss_t = eta_t
    m_out = _backward_message((tree_add(eta_in, eta_s_t), scale_in), eta_ss_t)
    return m_out, m_out

def dds_marginals(natparams):
    mnull = jnp.zeros_like(first(natparams[0]))
    mfwd = tree_prepend(
        scan(lambda m, _: (_forward_message((m[0] + _[0], m[1]), _[1]),) * 2, (mnull, .0),
             xs=(droplast(natparams[0]), natparams[1]))[1],
        (mnull, .0))[0]
    mbwd = tree_append(
        scan(lambda m, _: (_backward_message((m[0] + _[0], m[1]), _[1]),) * 2, (mnull, .0),
             xs=(dropfirst(natparams[0]), natparams[1]), reverse=True)[1],
        (mnull, .0))[0]
    qs = mfwd + mbwd + natparams[0]
    qss = vmap(_join_natparams)(
        droplast(mfwd + natparams[0]),
        dropfirst(mbwd + natparams[0]),
        natparams[1])
    return DdsMarginals(qs, qss)

def dds_stats(x, K):
    ds = jax.nn.one_hot(x, K) # [..., T, K]
    dss = ds[...,:-1,:,None] * ds[...,1:,None,:]
    return ds, dss

def dds_logprob(natparams, x):
    assert natparams[0].shape[:-1] == x.shape
    K = natparams[0].shape[-1]
    return dds_dot(dds_stats(x, K), natparams) - dds_logZ(natparams)

def dds_sample(key, natparams):
    assert(natparams[0].ndim == 2)
    qs, qss = dds_marginals(natparams)
    T = qs.shape[0]
    key = split(key, T)
    s0 = discrete_sample(key[0], qs[...,0,:])
    def _step(s_in, _):
        k, qss_t = _
        qs_out = discrete_condition(qss_t, _maskfirst(), s_in)
        s_out = discrete_sample(k, qs_out)
        return s_out, s_out    
    st = scan(_step, s0, (key[1:], qss))[1]
    # discrete distribution has trailing singleton dimension in 1d case
    return tree_prepend(st, s0).squeeze(-1)

# p(s0 = i) = a0[i]
# p(s_{t+1} = j | s_t = i) = A[t,i,j]
def dds_natparams_from_standard(standardparams):
    A, a0 = standardparams
    T = A.shape[-3] + 1
    singleton_natparams = jnp.pad(jnp.log(a0)[...,None,:], [(0,0)]*(a0.ndim-1) + [(0,T-1), (0,0)])
    pairwise_natparams = jnp.log(A)
    return singleton_natparams, pairwise_natparams

def dds_meanparams(natparams):
    return tuple(map(vmap(discrete_meanparams), dds_marginals(natparams)))

def dds_dot(natparams, stats):
    return jnp.sum(vmap(discrete_dot)(natparams[0], stats[0]), 0) \
        + jnp.sum(vmap(discrete_dot)(natparams[1], stats[1]), 0)

def dds_entropy(natparams):
    return dds_logZ(natparams) - dds_dot(natparams, dds_meanparams(natparams))

def dds_logZ(natparams):
    mnull = tree_map(jnp.zeros_like, first(natparams[0]))
    mfwd, logZ = scan(lambda m, _: (_forward_message((tree_add(m[0], _[0]), m[1]), _[1]),)*2, (mnull, .0),
        xs=(droplast(natparams[0]), natparams[1]))[0]
    return logZ + discrete_logZ(mfwd + natparams[0][-1])