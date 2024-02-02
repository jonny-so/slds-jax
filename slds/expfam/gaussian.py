import jax
import jax.numpy as jnp
from jax import Array
from slds.util import invcholp, outer, submatrix, vdot, mvp, transpose, tree_sub, tree_scale, tree_map

GaussianNatparams = tuple[Array, Array]
GaussianMeanparams = tuple[Array, Array]

# marginalise out masked dimensions
def gaussian_marginalise(natparams, mask):
    Jaa = submatrix(natparams[1], mask, mask)
    Jab = submatrix(natparams[1], mask, ~mask)
    Jbb = submatrix(natparams[1], ~mask, ~mask)
    ha, hb = natparams[0][mask], natparams[0][~mask]
    L = jnp.linalg.cholesky(-2*Jaa)
    v = jax.scipy.linalg.solve_triangular(L, ha, lower=True)
    M = jax.scipy.linalg.solve_triangular(L, -2*Jab, lower=True)
    J = Jbb + .5*jnp.matmul(M.T, M)
    h = hb + 2*Jab.T.dot(jax.scipy.linalg.solve_triangular(L.T, v, lower=False))
    logZ = .5*h.shape[-1]*jnp.log(2*jnp.pi) + .5*vdot(v,v) - jnp.sum(jnp.log(jnp.diag(L)),-1)
    return (h, J), logZ

# condition on masked dimensions
def gaussian_condition(natparams, mask, x):
    Jab = submatrix(natparams[1], mask, ~mask)
    Jbb = submatrix(natparams[1], ~mask, ~mask)
    hb = natparams[0][~mask]
    J = Jbb
    h = hb + 2*Jab.T.dot(x)
    return h, J

def gaussian_condition_meanparams(meanparams, mask, x):
    Vall = meanparams[1] - outer(meanparams[0], meanparams[0])
    Vab = submatrix(Vall, mask, ~mask)
    Vaa = submatrix(Vall, mask, mask)
    Vbb = submatrix(Vall, ~mask, ~mask)
    mu_a = meanparams[0][mask]
    mu_b = meanparams[0][~mask]
    L = jnp.linalg.cholesky(Vaa)
    V = Vbb - jnp.matmul(transpose(Vab), invcholp(L, Vab))
    mu = mu_b + jnp.matmul(transpose(Vab), invcholp(L, x-mu_a))
    return mu, V + outer(mu, mu)

def gaussian_sample(key, natparams, shape=()):
    expand = lambda _: jnp.tile(_, shape + (1,)*_.ndim)
    natparams = tree_map(expand, natparams)
    h, J = natparams
    V = jnp.linalg.inv(-2*J)
    mu = mvp(V, h)
    return jax.random.multivariate_normal(key, mu, V)

def gaussian_sample_meanparams(rng, meanparams):
    mu, V = meanparams[0], meanparams[1] - outer(meanparams[0], meanparams[0])
    return jax.random.multivariate_normal(rng, mu, V, method='eigh')

def gaussian_natparams(meanparams):
    x, xx = meanparams
    J = -.5*jnp.linalg.inv(xx - outer(x,x))
    h = -2*mvp(J, x)
    return h, J

def gaussian_meanparams(natparams):
    h, J = natparams
    J = .5*(J + transpose(J))
    V = jnp.linalg.inv(-2*J)
    mu = mvp(V, h)
    return mu, V + outer(mu, mu)

def gaussian_stats(x):
    return x, outer(x, x)

def gaussian_standardparams(natparams):
    h, J = natparams
    V = jnp.linalg.inv(-2*J)
    mu = mvp(V, h)
    return mu, V

def gaussian_mean(natparams):
    return gaussian_standardparams(natparams)[0]

def gaussian_var(natparams):
    return gaussian_standardparams(natparams)[1]

def gaussian_logZ(natparams, jitter=.0):
    h, J = natparams
    D = h.shape[-1]
    J = .5*(J + transpose(J))
    L = jnp.linalg.cholesky(-2*J + jnp.eye(D)*jitter)
    v = jax.scipy.linalg.solve_triangular(L, h, lower=True)
    halflogdet = jnp.sum(jnp.log(jnp.diagonal(L, axis1=-1, axis2=-2)), -1)
    return .5*h.shape[-1]*jnp.log(2*jnp.pi) + .5*vdot(v, v) - halflogdet

def gaussian_logprob(natparams, x):
    return gaussian_dot(natparams, gaussian_stats(x)) - gaussian_logZ(natparams)

def gaussian_entropy(natparams):
    h, J = natparams
    D = h.shape[-1]
    return .5*D*(1 + jnp.log(2*jnp.pi)) - .5*jnp.linalg.slogdet(-2*J)[1]

def gaussian_entropy_meanparams(meanparams):
    x, xx = meanparams
    D = x.shape[-1]
    V = xx - outer(x,x)
    return .5*D*(1 + jnp.log(2*jnp.pi)) + .5*jnp.linalg.slogdet(V)[1]

def gaussian_expand(natparams, mask):
    D = len(mask)
    h = jnp.zeros(D).at[mask].set(natparams[0])
    J = jnp.zeros((D,D)).at[outer(mask,mask)].set(natparams[1].reshape(-1))
    return h, J

def gaussian_dot(natparams, stats):
    h, J = natparams
    x, xx = stats
    return jnp.sum(h*x, axis=-1) + jnp.sum(J*xx, axis=(-1,-2))

def gaussian_kl(natparams1, natparams2):
    return gaussian_logZ(natparams2) - gaussian_logZ(natparams1) \
        + gaussian_dot(tree_sub(natparams1, natparams2), gaussian_meanparams(natparams1))

def gaussian_kl_meanparams(meanparams1, meanparams2):
    natparams1, natparams2 = gaussian_natparams(meanparams1), gaussian_natparams(meanparams2)
    return gaussian_logZ(natparams2) - gaussian_logZ(natparams1) \
        + gaussian_dot(tree_sub(natparams1, natparams2), meanparams1)

def gaussian_symmetrize(_):
    return _[0], .5*(transpose(_[1]) + _[1])
