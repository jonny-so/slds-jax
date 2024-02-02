# linear dynamical system:
#   z_0 ~ N(b_0; inv(Q0))
#   z_t ~ N(B z_{t-1} + b; inv(Q))

import jax.numpy as jnp
import numpy
from jax import vmap
from jax.lax import scan
from jax.random import split
from typing import NamedTuple

from slds.expfam.gaussian import *
from slds.util import *

LdsNatparams = tuple[GaussianNatparams, GaussianNatparams]
LdsMeanparams = tuple[GaussianMeanparams, GaussianMeanparams]

class LdsMarginals(NamedTuple):
    singletons: GaussianNatparams
    pairwise: GaussianNatparams

_maskfirst = lambda n,d: numpy.arange(d) < n
_masklast = lambda n,d: numpy.arange(d) >= d-n

def _join_natparams(eta_a, eta_b, eta_ab):
    Da, Db = len(eta_a[0]), len(eta_b[0])
    eta_a = gaussian_expand(eta_a, _maskfirst(Da,Da+Db))
    eta_b = gaussian_expand(eta_b, _masklast(Db,Da+Db))
    return tree_sum([eta_a, eta_b, eta_ab])

def _forward_message(mfwd, eta_transition):
    mask = _maskfirst(len(mfwd[0][0]), len(eta_transition[0]))
    m, logZ = gaussian_marginalise(tree_add(gaussian_expand(mfwd[0], mask), eta_transition), mask)
    return m, logZ + mfwd[1]

def _backward_message(mbwd, eta_transition):
    mask = _masklast(len(mbwd[0][0]), len(eta_transition[0]))
    m, logZ = gaussian_marginalise(tree_add(gaussian_expand(mbwd[0], mask), eta_transition), mask)
    return m, logZ + mbwd[1]

def _forward_step(m_in, eta_t):
    eta_in, scale_in = m_in
    eta_z_t, eta_zz_t = eta_t
    m_out = _forward_message((tree_add(eta_in, eta_z_t), scale_in), eta_zz_t)
    return m_out, m_out
    
def _backward_step(m_in, eta_t):
    eta_in, scale_in = m_in
    eta_z_t, eta_zz_t = eta_t
    m_out = _backward_message((tree_add(eta_in, eta_z_t), scale_in), eta_zz_t)
    return m_out, m_out

# return marginal gaussian natparams for singleton and (neighbouring) pairwise cliques
def lds_marginals(natparams):
    mnull = tree_map(jnp.zeros_like, tree_first(natparams[0]))
    eta_fwd = tree_prepend(
        scan(_forward_step, init=(mnull, .0),
            xs=(tree_droplast(natparams[0]), natparams[1]))[1],
        (mnull, .0))[0]
    eta_bwd = tree_append(
        scan(_backward_step, init=(mnull, .0),
             xs=(tree_dropfirst(natparams[0]), natparams[1]), reverse=True)[1],
        (mnull, .0))[0]
    qz = tree_sum([eta_fwd, eta_bwd, natparams[0]])
    qzz = vmap(_join_natparams)(
        tree_droplast(tree_sum([eta_fwd, natparams[0]])),
        tree_dropfirst(tree_sum([eta_bwd, natparams[0]])),
        natparams[1])
    return LdsMarginals(qz, qzz)

def lds_meanparams(natparams):
    return tuple(map(vmap(gaussian_meanparams), lds_marginals(natparams)))

def lds_stats(z):
    stats_z = gaussian_stats(z)
    stats_zz = gaussian_stats(jnp.concatenate([z[..., :-1, :], z[..., 1:, :]], -1))
    return stats_z, stats_zz

def lds_logZ(natparams):
    mnull = tree_map(jnp.zeros_like, tree_first(natparams[0]))
    mfwd, logZ = scan(_forward_step, init=(mnull, .0),
        xs=(tree_droplast(natparams[0]), natparams[1]))[0]
    return logZ + gaussian_logZ(tree_add(mfwd, tree_last(natparams[0])))

def lds_entropy(natparams):
    return lds_logZ(natparams) - lds_dot(natparams, lds_meanparams(natparams))

def lds_dot(eta, stats):
    return jnp.sum(vmap(gaussian_dot)(eta[0], stats[0]), 0) \
        + jnp.sum(vmap(gaussian_dot)(eta[1], stats[1]), 0)

def lds_logprob(natparams, z):
    return lds_dot(natparams, lds_stats(z)) - lds_logZ(natparams)

def lds_sample(key, natparams):
    assert(natparams[0][0].ndim == 2)
    qz, qzz = lds_marginals(natparams)
    T, D = qz[0].shape
    key = split(key, T)
    z0 = gaussian_sample(key[0], tree_first(qz))
    def _step(z_in, _):
        k, qzz_t = _
        qz_out = gaussian_condition(qzz_t, _maskfirst(D, 2*D), z_in)
        z_out = gaussian_sample(k, qz_out)
        return z_out, z_out
    zt = scan(_step, z0, (key[1:], qzz))[1]
    return tree_prepend(zt, z0)

# natparams singleton factor N(z_0; b_0, inv(Q0))
def _prior_natparams(b0, Q0):
    J = -.5*Q0
    h = Q0.dot(b0)
    return h, J

# natparams for pairwise factor N(z_2; A z_1 + a; inv(Q)), with layout [z_1; z_2]
def _transition_natparams(B, b, Q):
    QB = jnp.matmul(Q, B)
    Jaa, Jab, Jbb = -.5*jnp.matmul(B.T, QB), .5*QB.T, -.5*Q
    J = jnp.block([[Jaa, Jab], [Jab.T, Jbb]])
    h = jnp.concatenate([-QB.T.dot(b), Q.dot(b)])
    return h, J

def lds_natparams_fromstandard(standardparams):
    B, b, b0, Q, Q0 = standardparams
    T = B.shape[0] + 1
    prior_natparams = _prior_natparams(b0, Q0)
    singleton_natparams = tree_stack([prior_natparams] + [tree_scale(prior_natparams, .0)]*(T-1))
    pairwise_natparams = vmap(_transition_natparams)(B, b, Q)
    return singleton_natparams, pairwise_natparams
