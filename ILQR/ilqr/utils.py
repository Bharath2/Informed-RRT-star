import sympy as sp
import numpy as np

from numba import njit


def GetSyms(n_x, n_u):
  '''
      Returns matrices with symbolic variables for states and actions
      n_x: state size
      n_u: action size
  '''

  x = sp.IndexedBase('x')
  u = sp.IndexedBase('u')
  xs = sp.Matrix([x[i] for i in range(n_x)])
  us = sp.Matrix([u[i] for i in range(n_u)])
  return xs, us


@njit
def FiniteDiff(fun, x, u, i, eps):
  '''
     Finite difference approximation
  '''

  args = (x, u)
  fun0 = fun(x, u)

  m = arg[0].size
  n = arg[i].size

  Jac = np.zeros((m, n))
  for k in range(n):
    args[i][k] += eps
    Jac[:, k] = (fun(args[0], args[1]) - fun0)/eps
    args[i][k] -= eps

  return Jac



def sympy_to_numba(f, *args):
    '''
       Converts sympy matrix or expression to numba jitted function
    '''
    if isinstance(f, sp.Matrix):
        #To convert all elements to floats
        m, n = f.shape
        f += 1e-8*np.ones((m, n))

        #To eleminate extra dimension
        if n == 1 or m == 1:
            f = sp.Array(f)
            if n == 1: f = f[:, 0]
            if m == 1: f = f[0, :]
            f = njit(sp.lambdify([*args], f, modules = 'numpy'))
            f_new = lambda *args: np.array(f(*args))
            return njit(f_new)

    f = sp.lambdify([*args], f, modules = 'numpy')
    return njit(f)
