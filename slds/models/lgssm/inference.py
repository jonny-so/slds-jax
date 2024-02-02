import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import random
import numpy
from jax import vmap
from jax.lax import scan
from jax.random import split
from typing import NamedTuple, Union

from slds.expfam.gaussian import *
from slds.expfam.lds import *
from slds.models.lgssm.model import *
from slds.util import *

_maskfirst = lambda n,d: numpy.arange(d) < n
_masklast = lambda n,d: numpy.arange(d) >= d-n

# prior factor for z0 ~ N(b0; inv(Q0))
def _prior_natparams(b0, Q0):
    J = -.5*Q0
    h = Q0.dot(b0)
    return h, J

def _prior_scale(b0, Q0):
    return -.5*b0.shape[-1]*jnp.log(2*jnp.pi) + .5*jnp.linalg.slogdet(Q0)[1] - .5*b0.T.dot(Q0).dot(b0)

# transition factor for z2 ~ N(A z1 + a; inv(Q)), with layout [z1; z2]
def _transition_natparams(B, b, Q):
    QB = jnp.matmul(Q, B)
    Jaa, Jab, Jbb = -.5*jnp.matmul(B.T, QB), .5*QB.T, -.5*Q
    J = jnp.block([[Jaa, Jab], [Jab.T, Jbb]])
    h = jnp.concatenate([-QB.T.dot(b), Q.dot(b)])
    return h, J

def _transition_scale(b, Q):
    return -.5*b.shape[-1]*jnp.log(2*jnp.pi) + .5*jnp.linalg.slogdet(Q)[1] - .5*b.T.dot(Q).dot(b)

# likelihood factor for x ~ N(Cz + c; inv(R))
def _likelihood_natparams(C, c, R, x):
    RC = jnp.matmul(R, C)
    J = -.5*jnp.matmul(C.T, RC)
    h = RC.T.dot(x - c)
    return h, J

def _likelihood_scale(c, R, x):
    return -.5*c.shape[-1]*jnp.log(2*jnp.pi) + .5*jnp.linalg.slogdet(R)[1] - .5*(x-c).T.dot(R).dot(x-c)

def _joint_natparams(model, observations):
    B, b, b0, C, c, Q, Q0, R = model
    T = observations.shape[0]
    eta_prior = _prior_natparams(b0, Q0)
    eta_likelihood = vmap(partial(_likelihood_natparams, C, c, R))(observations)
    eta_z = tree_prepend(
        tree_dropfirst(eta_likelihood),
        tree_add(eta_prior, tree_first(eta_likelihood)))
    eta_zz = vmap(lambda _: _transition_natparams(B, b, Q))(jnp.arange(T-1))
    return eta_z, eta_zz

def _joint_scale(model, observations):
    _, b, b0, _, c, Q, Q0, R = model
    T = observations.shape[0]
    scale_prior = _prior_scale(b0, Q0)
    scale_likelihood = vmap(lambda _: _likelihood_scale(c, R, _))(observations)
    scale_z = scale_likelihood.at[0].add(scale_prior)
    scale_zz = vmap(lambda _: _transition_scale(b, Q))(jnp.arange(T-1))
    return scale_z, scale_zz

class LgssmPosterior(NamedTuple):
    marginals: LdsMarginals

class LgssmInferenceState(NamedTuple):
    joint: LdsNatparams
    marginals: Union[None, LgssmPosterior]

def lgssm_inference_init(model: Lgssm, observations: Array):
    joint = _joint_natparams(model, observations)
    return LgssmInferenceState(joint, None)

def lgssm_inference_update(state: LgssmInferenceState):
    posterior = LgssmPosterior(lds_marginals(state.joint))
    return LgssmInferenceState(state.joint, posterior)

def lgssm_inference_posterior(state: LgssmInferenceState):
    return state.marginals