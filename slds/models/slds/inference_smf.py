import jax.numpy as jnp
from slds.expfam.dds import *
from slds.expfam.discrete import *
from slds.expfam.gaussian import *
from slds.expfam.lds import *
from slds.models.hmm.inference import *
from slds.models.lgssm.model import *
from slds.models.slds.model import *
import slds.models.lgssm.inference as lgssm_inference
from jax import vmap
from jax.experimental.host_callback import id_print

# reduce a batch of gaussian natparams to a single gaussian natparams using weights w
def _gaussian_reduce_natparams(natparams, w):
    return (w[:,None] * natparams[0]).sum(0), (w[:,None, None]*natparams[1]).sum(0)

def _update_discrete_natparams(joint, continuous_natparams):

    singleton_meanparams, pairwise_meanparams = lds_meanparams(continuous_natparams)

    singleton_scales, pairwise_scales = joint.continuous_scales
    lds_message = singleton_scales.T + vmap(
            gaussian_dot, in_axes=(1,0)
        )(joint.continuous_natparams[0], singleton_meanparams)
    lds_message = lds_message.at[1:].add(pairwise_scales.T + vmap(
            gaussian_dot, in_axes=(1,0)
        )(joint.continuous_natparams[1], pairwise_meanparams))

    # lds message only affects singleton factors
    singleton_natparams = joint.discrete_natparams[0] + lds_message
    pairwise_natparams = joint.discrete_natparams[1]
    natparams = singleton_natparams, pairwise_natparams

    return natparams

def _update_continuous_natparams(joint, discrete_natparams):

    discrete_probabilities = dds_meanparams(discrete_natparams)[0]

    singleton_natparams = vmap(
            _gaussian_reduce_natparams, in_axes=(1,0)
        )(joint.continuous_natparams[0], discrete_probabilities)
    pairwise_natparams = vmap(
            _gaussian_reduce_natparams, in_axes=(1,0)
        )(joint.continuous_natparams[1], discrete_probabilities[1:])
    natparams = singleton_natparams, pairwise_natparams
    
    return natparams

class SldsSmfJoint(NamedTuple):
    discrete_natparams: DdsNatparams
    continuous_natparams: LdsNatparams
    continuous_scales: tuple[Array, Array]

class SldsSmfPosterior(NamedTuple):
    discrete_natparams: DdsNatparams
    continuous_natparams: LdsNatparams

class SldsSmfInferenceState(NamedTuple):
    joint: SldsSmfJoint
    posterior: SldsSmfPosterior

def _init_joint(model, observations):
    A, a0, B, b, b0, C, c, Q, Q0, R = model

    T = observations.shape[0]
    discrete_params = jnp.stack([A]*(T-1)), a0
    discrete_natparams = dds_natparams_from_standard(discrete_params)

    lgssm_params = B, b, b0, C, c, Q, Q0, R
    continuous_natparams = vmap(
        lambda _: lgssm_inference._joint_natparams(Lgssm(*_), observations)
    )(lgssm_params)
    continuous_scales = vmap(
        lambda _: lgssm_inference._joint_scale(Lgssm(*_), observations)
    )(lgssm_params)
    
    return SldsSmfJoint(discrete_natparams, continuous_natparams, continuous_scales)

def _init_posterior(joint):
    discrete_natparams = joint.discrete_natparams
    continuous_natparams = tree_map(partial(jnp.mean, axis=0), joint.continuous_natparams)
    return SldsSmfPosterior(discrete_natparams, continuous_natparams)

def slds_smf_inference_init(model: Slds, observations: Array):
    joint = _init_joint(model, observations)
    posterior = _init_posterior(joint)
    return SldsSmfInferenceState(joint, posterior)

def slds_smf_inference_update(state: SldsSmfInferenceState):
    continuous_natparams = _update_continuous_natparams(state.joint, state.posterior.discrete_natparams)
    discrete_natparams = _update_discrete_natparams(state.joint, state.posterior.continuous_natparams)
    posterior = SldsSmfPosterior(discrete_natparams, continuous_natparams)
    return SldsSmfInferenceState(state.joint, posterior)

def slds_smf_inference_posterior(state: SldsSmfInferenceState):
    return state.posterior

def slds_smf_elbo(inference_state: SldsSmfInferenceState):

    joint, posterior = inference_state

    discrete_meanparams = dds_meanparams(posterior.discrete_natparams)
    continuous_meanparams = lds_meanparams(posterior.continuous_natparams)
    discrete_probabilities = discrete_meanparams[0]
    
    lds_singleton_scales, lds_pairwise_scales = joint.continuous_scales
    lds_singleton_natparams, lds_pairwise_natparams = joint.continuous_natparams
    
    lds_singleton_terms = lds_singleton_scales.T + vmap(
            gaussian_dot, in_axes=(1, 0)
        )(lds_singleton_natparams, continuous_meanparams[0])
    lds_pairwise_terms = lds_pairwise_scales.T + vmap(
            gaussian_dot, in_axes=(1, 0)
        )(lds_pairwise_natparams, continuous_meanparams[1])
    
    # reduce using discrete state probabilities
    E_logpxz_s = (jnp.sum(discrete_probabilities*lds_singleton_terms)
        + jnp.sum(discrete_probabilities[1:]*lds_pairwise_terms))
    
    E_logps = dds_dot(joint.discrete_natparams, discrete_meanparams) - dds_logZ(joint.discrete_natparams)
    E_logqs = dds_dot(posterior.discrete_natparams, discrete_meanparams) - dds_logZ(posterior.discrete_natparams)
    E_logqz = lds_dot(posterior.continuous_natparams, continuous_meanparams) - lds_logZ(posterior.continuous_natparams)

    return E_logpxz_s + E_logps - E_logqs - E_logqz
