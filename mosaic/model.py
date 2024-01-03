import jax
import jax.numpy as np


@jax.custom_jvp
def gr_than(x, thr):
    """ Thresholding function for spiking neurons. """
    return (x > thr).astype(np.float32)


@gr_than.defjvp
def gr_jvp(primals, tangents):
    """ Surrogate gradient function for thresholding. """
    x, thr = primals
    x_dot, y_dot = tangents
    primal_out = gr_than(x, thr)
    tangent_out = x_dot / (10 * np.absolute(x - thr) + 1)**2
    return primal_out, tangent_out


def lif_forward(state, x):
    """ Inference for Leaky Integrate and Fire (LIF)-based 
        tiled recurrent neural network architecture.

        Input is fed to the first row of neuron tiles (North), 
        later output will be readout from the last row of tiles (South).
    """
    rec_weight = state[0]  # Static weights
    thr_rec, alpha, inp_mask, no_rec_mask = state[1]  # Static neuron states
    v, z = state[2]  # Dynamic neuron states

    inp_dim = inp_mask.shape[0]
    rec_dim = no_rec_mask.shape[0]

    I = np.pad(
        np.matmul(x, rec_weight[0:inp_dim, 0:inp_dim] * inp_mask),
        (0, rec_dim-inp_dim), 'constant')
    v = alpha * v + I + np.matmul(z, rec_weight * no_rec_mask) - z * thr_rec
    z = gr_than(v, thr_rec)

    return [rec_weight, [thr_rec, alpha, inp_mask, no_rec_mask], [v, z]], z