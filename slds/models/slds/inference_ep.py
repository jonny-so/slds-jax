import jax.numpy as jnp
from jax.scipy.special import logsumexp
from slds.expfam.discrete import *
from slds.expfam.gaussian import *
from slds.expfam.gmm import *
from slds.models.hmm.inference import *
from slds.models.lgssm.model import *
from slds.models.slds.model import *
from jax import vmap

_maskfirst = lambda n,d: numpy.arange(d) < n
_masklast = lambda n,d: numpy.arange(d) >= d-n

# numerically stable representation of meanparams
def _gmm_stableparams(natparams):
    gamma, h, J = natparams
    J = .5*(J + transpose(J))
    logZ = gaussian_logZ((h, J))
    logp = gamma + logZ - logsumexp(gamma + logZ)
    m = gaussian_meanparams((h, J))
    return logp, m[0], m[1]

def _gmm_natparams_from_stable(stableparams):
    logp, x, xx = stableparams
    h, J = gaussian_natparams((x, xx))
    gamma = logp - gaussian_logZ((h, J))
    return gamma, h, J

def _gmm_collapse_stable(natparams, axis):
    assert(axis >= 0)
    logp, x, xx = _gmm_stableparams(natparams)
    logp_sum = logsumexp(logp, axis, keepdims=True)
    x = jnp.sum(jnp.exp(logp - logp_sum)[...,None]*x, axis)
    xx = jnp.sum(jnp.exp(logp - logp_sum)[...,None, None]*xx, axis)
    logp = logp_sum.squeeze(axis)
    return logp, x, xx

def _prior_natparams(a0, b0, Q0):
    J = -.5*Q0
    h = mvp(Q0, b0)
    gamma = jnp.log(a0) - gaussian_logZ((h, J))
    return gamma, h, J

def _transition_natparams(A, B, b, Q):
    K = A.shape[0]
    T = transpose
    QB = jnp.matmul(Q, B)
    Jaa, Jab, Jbb = -.5*jnp.matmul(T(B), QB), .5*transpose(QB), -.5*Q
    J = jnp.tile(jnp.block([[Jaa, Jab], [T(Jab), Jbb]]), (K, 1, 1, 1))
    h = jnp.tile(jnp.concatenate([-mvp(T(QB), b), mvp(Q, b)], -1), (K, 1, 1))
    scale = -.5*b.shape[-1]*jnp.log(2*jnp.pi) + .5*jnp.linalg.slogdet(Q)[1] - .5*vdot(b, mvp(Q,b))
    gamma = jnp.log(A) + scale[None,:]
    return gamma, h, J

def _likelihood_natparams(C, c, R, x):
    T = transpose
    RC = jnp.matmul(R, C)
    J = -.5*jnp.matmul(T(C), RC)
    h = mvp(T(RC), x - c)
    scale = -.5*c.shape[-1]*jnp.log(2*jnp.pi) + .5*jnp.linalg.slogdet(R)[1] - .5*vdot(x-c, mvp(R, x-c))
    gamma = scale
    return gamma, h, J

def _tilted_natparams(eta_cavity, eta_transition):
    D1, D2 = eta_cavity[0][1].shape[1], eta_cavity[1][1].shape[1]
    eta_cavity_fwd = (eta_cavity[0][0], *vmap(gaussian_expand, in_axes=(0,None))(eta_cavity[0][1:], _maskfirst(D1, D1+D2)))
    eta_cavity_bwd = (eta_cavity[1][0], *vmap(gaussian_expand, in_axes=(0,None))(eta_cavity[1][1:], _masklast(D2, D1+D2)))
    gamma = eta_transition[0] + eta_cavity_fwd[0][:,None] + eta_cavity_bwd[0][None,:]
    h = eta_transition[1] + eta_cavity_fwd[1][:,None] + eta_cavity_bwd[1][None,:]
    J = eta_transition[2] + eta_cavity_fwd[2][:,None] + eta_cavity_bwd[2][None,:]
    return gamma, h, J

def _cavity_natparams(mfwd, mbwd, eta_likelihood1, eta_likelihood2):
    eta_left = tree_add(mfwd, eta_likelihood1)
    eta_right = tree_add(mbwd, eta_likelihood2)
    return eta_left, eta_right

def _project_fwd(eta_tilted, D1, D2):
    logp, x, xx = _gmm_collapse_stable(eta_tilted, 0)
    mask = _masklast(D2, D1+D2)
    x = x[:, mask]
    xx = vmap(lambda _: submatrix(_, mask, mask))(xx)
    return _gmm_natparams_from_stable((logp, x, xx))

def _project_bwd(eta_tilted, D1, D2):
    logp, x, xx = _gmm_collapse_stable(eta_tilted, 1)
    mask = _maskfirst(D1, D1+D2)
    x = x[:, mask]
    xx = vmap(lambda _: submatrix(_, mask, mask))(xx)
    return _gmm_natparams_from_stable((logp, x, xx)), jnp.linalg.slogdet(eta_tilted[2])[0]

def _forward_message(mfwd, mbwd, mlast, eta_transition, eta_likelihood1, eta_likelihood2, damp):
    D1, D2 = mfwd[1].shape[1], mbwd[1].shape[1]
    eta_cavity = _cavity_natparams(mfwd, mbwd, eta_likelihood1, eta_likelihood2)
    eta_tilted = _tilted_natparams(eta_cavity, eta_transition)
    mnext = tree_sub(_project_fwd(eta_tilted, D1, D2), eta_cavity[1]) # right site param
    return tree_interpolate(damp, mlast, mnext)

def _backward_message(mfwd, mbwd, mlast, eta_transition, eta_likelihood1, eta_likelihood2, damp):
    D1, D2 = mfwd[1].shape[1], mbwd[1].shape[1]
    eta_cavity = _cavity_natparams(mfwd, mbwd, eta_likelihood1, eta_likelihood2)
    eta_tilted = _tilted_natparams(eta_cavity, eta_transition)
    mnext = tree_sub(_project_bwd(eta_tilted, D1, D2)[0], eta_cavity[0]) # left site param
    return tree_interpolate(damp, mlast, mnext)

def _update_forward_messages(state, damping):
    joint = state.joint
    backward_messages = tree_dropfirst(state.backward_messages)
    old_forward_messages = tree_dropfirst(state.forward_messages)
    left_likelihoods = tree_droplast(joint.eta_likelihood)
    right_likelihoods = tree_dropfirst(joint.eta_likelihood)
    forward_messages = tree_prepend(
        scan(lambda m, _: (_forward_message(m, *_, damping),)*2, joint.eta_prior,
            (backward_messages, old_forward_messages, joint.eta_transition, left_likelihoods, right_likelihoods),
            reverse=False)[1],
        joint.eta_prior)
    forward_messages = gmm_symmetrize(forward_messages)
    return SldsEpInferenceState(joint, forward_messages, state.backward_messages)

def _update_backward_messages(state, damping):
    joint = state.joint
    forward_messages = tree_droplast(state.forward_messages)
    old_backward_messages = tree_droplast(state.backward_messages)
    left_likelihoods = tree_droplast(joint.eta_likelihood)
    right_likelihoods = tree_dropfirst(joint.eta_likelihood)
    eta_null = tree_map(jnp.zeros_like, joint.eta_prior)
    backward_messages = tree_append(
        scan(lambda m, _: (_backward_message(_[0], m, *_[1:], damping),)*2, eta_null,
            (forward_messages, old_backward_messages, joint.eta_transition, left_likelihoods, right_likelihoods),
            reverse=True)[1],
        eta_null)
    backward_messages = gmm_symmetrize(backward_messages)    
    return SldsEpInferenceState(joint, state.forward_messages, backward_messages)

def _init_joint(model, observations):
    A, a0, B, b, b0, C, c, Q, Q0, R = model
    T = observations.shape[0]
    eta_prior = _prior_natparams(a0, b0, Q0)
    eta_transition = vmap(lambda _: _transition_natparams(A, B, b, Q))(jnp.arange(T-1))
    eta_likelihood = vmap(lambda _: _likelihood_natparams(C, c, R, _))(observations)
    return SldsEpJoint(eta_prior, eta_transition, eta_likelihood)

class SldsEpJoint(NamedTuple):
    eta_prior: GmmNatparams
    eta_transition: GmmNatparams
    eta_likelihood: GmmNatparams

class SldsEpPosterior(NamedTuple):
    singleton_marginals: GmmNatparams

class SldsEpInferenceState(NamedTuple):
    joint: SldsEpJoint
    forward_messages: GmmNatparams
    backward_messages: GmmNatparams

def slds_ep_inference_init(model: Slds, observations: Array):
    joint = _init_joint(model, observations)
    forward_messages = tree_map(jnp.zeros_like, joint.eta_likelihood)
    backward_messages = tree_map(jnp.zeros_like, joint.eta_likelihood)
    return SldsEpInferenceState(joint, forward_messages, backward_messages)

def slds_ep_inference_update(state: SldsEpInferenceState, damping: float):
    state = _update_forward_messages(state, damping)
    state = _update_backward_messages(state, damping)
    return state

def slds_ep_inference_posterior(state: SldsEpInferenceState):
    singleton_marginals = tree_sum([
        state.forward_messages,
        state.backward_messages,
        state.joint.eta_likelihood])
    return SldsEpPosterior(singleton_marginals)

def slds_ep_energy(state: SldsEpInferenceState):
    joint, forward_messages, backward_messages = state
    _, eta_transition, eta_likelihood = joint

    cavity_natparams = vmap(_cavity_natparams)(
        tree_droplast(forward_messages),
        tree_dropfirst(backward_messages),
        tree_droplast(eta_likelihood),
        tree_dropfirst(eta_likelihood))
    
    singleton_marginals = tree_sum([forward_messages, backward_messages, eta_likelihood])
    pairwise_pseudomarginals = vmap(_tilted_natparams)(cavity_natparams, eta_transition)

    return jnp.sum(vmap(gmm_logZ)(pairwise_pseudomarginals), 0) - jnp.sum(vmap(gmm_logZ)(singleton_marginals)[1:-1], 0)