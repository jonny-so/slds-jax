import jax
import jax.numpy as jnp
import numpy as np
import operator
from functools import partial
from jax.tree_util import tree_flatten, tree_map, tree_unflatten

# inv(L*L.T)*Y
def invcholp(L, Y):
    D = jax.scipy.linalg.solve_triangular(L, Y, lower=True)
    B = jax.scipy.linalg.solve_triangular(L.T, D, lower=False)
    return B

# inv(X)*Y
invmp = lambda X, Y: invcholp(jax.linalg.cholesky(X), Y)

# batched outer product
outer = lambda x, y: x[...,None]*y[...,None,:]

# batched transpose
transpose = lambda _: jnp.swapaxes(_, -1, -2)

# batched matrix vector / vector (transpose) matrix product
mvp = lambda X, v: jnp.matmul(X, v[...,None]).squeeze(-1)
vmp = lambda v, X: jnp.matmul(v[...,None,:], X).squeeze(-2)
mmp = jnp.matmul

# batched vector dot product
vdot = lambda x, y: jnp.sum(x*y, -1)

# batched symmetrize
symmetrize = lambda _: .5*(_ + transpose(_))

def submatrix(x, rowmask, colmask):
    return x[outer(rowmask,colmask)].reshape(np.sum(rowmask), np.sum(colmask))

# call rng function and return value with new rng
def rngcall(rng, f, *args, **kwargs):
    rng1, rng2 = jax.random.split(rng)
    return rng1, f(rng2, *args, **kwargs)

dropfirst = lambda _: _[1:]
droplast = lambda _: _[:-1]
first = lambda _: _[0]
last = lambda _: _[-1]

def append(xs, x, axis=0):
    return tree_cat([xs, jnp.expand_dims(x, axis)], axis)
def prepend(xs, x, axis=0):
    return tree_cat([jnp.expand_dims(x, axis), xs], axis)

# tree utilities
tree_dropfirst = lambda _: tree_map(lambda x: x[1:], _)
tree_droplast = lambda _: tree_map(lambda x: x[:-1], _)
tree_first = lambda _: tree_map(lambda x: x[0], _)
tree_last = lambda _: tree_map(lambda x: x[-1], _)

def tree_append(xs, x, axis=0):
    x = tree_map(lambda _: jnp.expand_dims(_, axis), x)
    return tree_cat([xs, x], axis)

def tree_prepend(xs, x, axis=0):
    x = tree_map(lambda _: jnp.expand_dims(_, axis), x)
    return tree_cat([x, xs], axis)

def tree_cat(trees, axis=0):
    flats, treedefs = tuple(zip(*list(map(tree_flatten, trees))))
    flats = tuple(zip(*flats))
    tree = list(map(partial(jnp.concatenate, axis=axis), flats))
    return tree_unflatten(treedefs[0], tree)

def tree_stack(trees, axis=0):
    flats, treedefs = tuple(zip(*list(map(tree_flatten, trees))))
    flats = tuple(zip(*flats))
    tree = list(map(partial(jnp.stack, axis=axis), flats))
    return tree_unflatten(treedefs[0], tree)

def tree_sum(trees):
    flats, treedefs = tuple(zip(*list(map(tree_flatten, trees))))
    flats = tuple(zip(*flats))
    sums = list(map(sum, flats))
    return tree_unflatten(treedefs[0], sums)

def tree_add(tree1, tree2):
    return tree_map(operator.add, tree1, tree2)

def tree_sub(tree1, tree2):
    return tree_map(operator.sub, tree1, tree2)

def tree_mul(tree1, tree2):
    return tree_map(operator.mul, tree1, tree2)

def tree_scale(tree, c):
    return tree_map(lambda _: c*_, tree)

def tree_interpolate(w, tree1, tree2):
    return tree_add(
        tree_scale(tree1, w), tree_scale(tree2, 1-w))

def tree_vec(tree, unvec=False):
    flat, treedef = tree_flatten(tree)
    shapes = list(map(jnp.shape, flat))
    lengths = list(map(np.prod, shapes))
    def _unvec(x):
        xs = np.split(x, np.cumsum(np.array(lengths[:-1])))
        flat = list(map(lambda _: _[0].reshape(_[1]), zip(xs, shapes)))
        return tree_unflatten(treedef, flat)
    x = jnp.concatenate(list(map(partial(jnp.reshape, newshape=-1), flat)))
    return (x, _unvec) if unvec else x

def isposdefh(h):
    return jax.numpy.linalg.eigh(h)[0][...,0] > 0