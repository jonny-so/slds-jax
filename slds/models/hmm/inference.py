import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import random
import numpy
from typing import NamedTuple, Union
from jax import Array, vmap
from jax.lax import scan

from slds.expfam.dds import *
from slds.expfam.discrete import *
from slds.models.hmm.model import Hmm
from slds.util import *

_maskfirst = lambda: np.array([True, False])
_masklast = lambda: np.array([False, True])

# prior factor natparams for p(s0 = i) = a0[i]
def _prior_natparams(a0):
    return jnp.log(a0)

# transition factor natparams for p(s_{t+1} = j | s_t = i) = A[t,i,j]
def _transition_natparams(A):
    return jnp.log(A)

# likelihood factor natparams for p(x = j | s = i) = C[i,j]
def _likelihood_natparams(C, x):
    return discrete_condition(jnp.log(C), _masklast(), x.reshape((1,)))

def _joint_natparams(model, observations):
    A, a0, C = model
    T = observations.shape[0]
    singleton_natparams = vmap(partial(_likelihood_natparams, C))(observations)
    singleton_natparams = singleton_natparams.at[0].add(_prior_natparams(a0))
    pairwise_natparams = vmap(lambda _: _transition_natparams(A))(jnp.arange(T-1))
    return singleton_natparams, pairwise_natparams

class HmmPosterior(NamedTuple):
    marginals: DdsMarginals

class HmmInferenceState(NamedTuple):
    joint: DdsNatparams
    marginals: Union[None, HmmPosterior]

def hmm_inference_init(model: Hmm, observations: Array):
    joint = _joint_natparams(model, observations)
    return HmmInferenceState(joint, None)

def hmm_inference_update(state: HmmInferenceState):
    posterior = HmmPosterior(dds_marginals(state.joint))
    return HmmInferenceState(state.joint, posterior)

def hmm_inference_posterior(state: HmmInferenceState):
    return state.marginals