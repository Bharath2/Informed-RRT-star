import numba
import numpy as np


class iLQR:

  def __init__(self, dynamics, cost):
    '''
       iterative Linear Quadratic Regulator
    '''
    self.cost = cost
    self.dynamics = dynamics
    self.params = {'regu_init' : 100, 'max_regu' : 10000, 'min_regu' : 0.01}

  def run(self, x0, us_init, maxiter = 50):
    return run_ilqr(self.dynamics.f, self.dynamics.f_prime, self.cost.L,
                    self.cost.Lf, self.cost.L_prime, self.cost.Lf_prime,
                    x0, us_init, max_iter = maxiter, **self.params)



@numba.njit
def run_ilqr(f, f_prime, L, Lf, L_prime, Lf_prime, x0, u_init, max_iter = 50,
             regu_init = 100, max_regu = 10000, min_regu = 0.01):
    '''
       iLQR main loop
    '''
    us = u_init
    regu = regu_init

    # First forward rollout
    xs, J_old = rollout(f, L, Lf, x0, us)

    # cost trace
    cost_trace = [J_old]

    # Run main loop
    for it in range(max_iter):

        # Backward and forward pass
        ks, Ks, expected_cost_redu = backward_pass(f_prime, L_prime, Lf_prime, xs, us, regu)
        xs_new, us_new, J_new = forward_pass(f, L, Lf, xs, us, ks, Ks)

        # Accept or reject iteration
        if J_old - J_new > 0:
            # Improvement! Accept new trajectories and lower regularization
            J_old  = J_new
            xs = xs_new
            us = us_new
            regu *= 0.7
            # Early termination if improvement is small
            if expected_cost_redu < 1e-5: break
        else:
            # Reject new trajectories and increase regularization
            regu *= 2.0

        cost_trace.append(J_old)
        regu = min(max(regu, min_regu), max_regu)

    return xs, us, cost_trace


@numba.njit
def rollout(f, L, Lf, x0, us):
    '''
      Rollout with initial state and action trajectory
    '''
    xs = np.zeros((us.shape[0] + 1, x0.shape[0]))
    xs[0] = x0
    cost = 0
    for n in range(1, us.shape[0] + 1):
      xs[n] = f(xs[n-1], us[n])
      cost += L(xs[n], us[n])

    cost += Lf(xs[-1])

    return xs, cost


@numba.njit
def forward_pass(f, L, Lf, xs, us, ks, Ks):
    '''
       Forward Pass
    '''
    xs_new = np.zeros(xs.shape)

    cost_new = 0.0
    xs_new[0] = xs[0]
    us_new = us + ks

    for n in range(us.shape[0]):
        us_new[n] += Ks[n].dot(xs_new[n] - xs[n])
        xs_new[n + 1] = f(xs_new[n], us_new[n])
        cost_new += L(xs_new[n], us_new[n])

    cost_new += Lf(xs_new[-1])

    return xs_new, us_new, cost_new


@numba.njit
def backward_pass(f_prime, L_prime, Lf_prime, xs, us, regu):
    '''
       Backward Pass
    '''
    ks = np.zeros(us.shape)
    Ks = np.zeros((us.shape[0], us.shape[1], xs.shape[1]))

    expected_cost_redu = 0
    V_x, V_xx = Lf_prime(xs[-1])
    for n in range(us.shape[0] - 1, -1, -1):

        f_x, f_u = f_prime(xs[n], us[n])
        l_x, l_u, l_xx, l_ux, l_uu  = L_prime(xs[n], us[n])

        #Q_terms
        Q_x  = l_x  + f_x.T@V_x
        Q_u  = l_u  + f_u.T@V_x
        Q_xx = l_xx + f_x.T@V_xx@f_x
        Q_ux = l_ux + f_u.T@V_xx@f_x
        Q_uu = l_uu + f_u.T@V_xx@f_u

        #gains
        Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0])*regu
        Q_uu_inv = np.linalg.inv(Q_uu_regu)

        k = -Q_uu_inv@Q_u
        K = -Q_uu_inv@Q_ux
        ks[n], Ks[n] = k, K

        #V_terms
        V_x = Q_x + K.T@Q_u + Q_ux.T@k + K.T@Q_uu@k
        V_xx = Q_xx + 2*K.T@Q_ux + K.T@Q_uu@K

        expected_cost_redu += expected_cost_reduction(Q_u, Q_uu, k)

    return ks, Ks, expected_cost_redu


@numba.njit
def expected_cost_reduction(Q_u, Q_uu, k):
    return -Q_u.T.dot(k) - 0.5 * k.T.dot(Q_uu.dot(k))
