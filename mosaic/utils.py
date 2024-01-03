import jax
import jax.numpy as np
import jax.random as random
from jax.nn import log_softmax
from functools import partial


def sparser(W_in, W_mask):
    """ 
    Calculate sparsity-inducing loss.
    
    This loss function encourages sparsity in the weights of a model 
    by penalizing the squared weights selectively, as determined 
    by a masking matrix.
    """
    L = np.mean(W_mask * W_in**2)
    return L


@partial(jax.jit, static_argnums=(1,2))
def pruner(train_step, prune_start_step, prune_thr, weight):
    """ Apply pruning to weights based on training step and threshold. """
    weight = jax.lax.cond(
        train_step > prune_start_step, 
        lambda w: np.where(np.abs(w) < prune_thr, 0., w),
        lambda w: w, 
        weight
    )

    weight = np.clip(weight, -2.5*np.std(weight), 2.5*np.std(weight))
    return weight


@jax.custom_jvp
def add_noise(weight, key, noise_std):
    """ Introduce additive conductance-based Gaussian noise 
        only for inference to simulate programming noise 
        due to analog nature of memristors.
    """
    weight = np.where(
        weight != 0.0, 
        weight + random.normal(key, weight.shape) * np.max(weight) * noise_std, 
        weight
    )

    weight = np.clip(weight, -1, 1)
    return weight
 

@add_noise.defjvp
def add_noise_jvp(primals, tangents):
    weight, key, noise_std = primals
    x_dot, _, _ = tangents
    primal_out = add_noise(weight, key, noise_std)
    tangent_out = x_dot
    return primal_out, tangent_out


def generate_mosaic_mask(n_rec=256, n_core=16, beta=0.4):
    """ Generate Mosaic mask for connectivity. 
    
    This function calculates the number of hops between any two neurons, based on 
    their physical proximity, i.e. number of hops required to communicate,
    in a tiled Mosaic architecture defined by `n_rec` and `n_core`. 
    Then it calculates a penalization mask based on the communication cost.
    """

    num_cols = np.sqrt(n_core)
    num_hops = np.zeros((n_core, n_core))

    for i in range(n_core):
        for j in range(i+1, n_core):
            i_col_id = i % num_cols
            i_row_id = i // num_cols
            j_col_id = j % num_cols
            j_row_id = j // num_cols

            x_hop = 0
            y_hop = 0
            # COLUMNS
            if i_col_id == j_col_id and np.abs(i_row_id - j_row_id) == 1:
                x_hop = 0
                y_hop = 1
            if i_col_id == j_col_id and np.abs(i_row_id - j_row_id) > 1:
                x_hop = 1
                y_hop = 2 * np.abs(i_row_id - j_row_id)
            if i_col_id != j_col_id and np.abs(i_col_id - j_col_id) == 1:
                x_hop = 1
                y_hop = 2 * np.abs(i_row_id - j_row_id)
            #Far away 
            if i_col_id != j_col_id and np.abs(i_col_id - j_col_id) > 1:
                x_hop = 2 * np.abs(i_col_id - j_col_id) - 1
                y_hop = 2 * np.abs(i_row_id - j_row_id)

            # ROWS
            if i_row_id == j_row_id and np.abs(i_col_id - j_col_id) == 1:
                x_hop = 1
                y_hop = 0
            if i_row_id == j_row_id and np.abs(i_col_id - j_col_id) > 1:
                x_hop = 1
                y_hop = 2 * np.abs(i_col_id - j_col_id)
            if i_row_id != j_row_id  and np.abs(i_row_id - j_row_id) == 1:
                x_hop = 2 * np.abs(i_col_id - j_col_id)
                y_hop = 1

            num_hops = num_hops.at[i,j].set(x_hop + y_hop)
            num_hops = num_hops.at[j,i].set(x_hop + y_hop)


    weight_mask = np.clip(
        np.exp(
            beta * np.kron(num_hops, np.ones((n_rec//n_core, n_rec//n_core)))
        ) - 1,
        0, np.exp(15)
    )
    return weight_mask


def calc_mosaic_stats(weights, mask, beta):
    """ Calculates percentage of occupancy of possible connections 
    for different number of hops between any neurons.
    """
    MAX_NUM_HOPS = int(np.max(np.log(mask)/beta))
    num_total_memristors_per_hop, _ = np.histogram(
        mask, np.exp(beta*np.arange(MAX_NUM_HOPS+2))
    )
    num_non_zero_weights_per_hop = np.array([
        np.sum(np.abs(weights[np.where(mask==np.exp(beta*i))])>0) 
        for i in range(MAX_NUM_HOPS+1)
    ])
    p_occ = 100 * num_non_zero_weights_per_hop / num_total_memristors_per_hop
    return np.nan_to_num(p_occ)


def fr_decoder(spikes):
    """ Calculates log-softmax based on spike count logits. """
    Yhat = log_softmax(np.sum(spikes, 1))
    return Yhat


def param_initializer(key, n_inp, n_rec, n_out, n_core, thr_rec, 
                      tau_rec, w_gain):
    """ Initialize parameters. """
    _, key_rec = random.split(key, 2)
    alpha = np.exp(-1e-3/tau_rec) 
    rec_weight = random.normal(key_rec, [n_rec, n_rec]) * w_gain
    num_neuron_per_tile = n_rec // n_core

    # No recurrent at input and output tiles
    no_rec_mask = np.ones((n_rec, n_rec))
    for i in range(int(n_inp//num_neuron_per_tile)):
        no_rec_mask = no_rec_mask.at[
            i*num_neuron_per_tile:(i+1)*num_neuron_per_tile,
            i*num_neuron_per_tile:(i+1)*num_neuron_per_tile
        ].set(0)

    no_rec_mask = no_rec_mask.at[-n_out:,-n_out:].set(0)

    # Set input weight matrix
    inp_mask = np.zeros((n_inp, n_inp))
    for i in range(int(n_inp//num_neuron_per_tile)):
        inp_mask = inp_mask.at[
            i*num_neuron_per_tile:(i+1)*num_neuron_per_tile,
            i*num_neuron_per_tile:(i+1)*num_neuron_per_tile
        ].set(1)

    neuron_dyn = [np.zeros(n_rec), np.zeros(n_rec)]
    net_params = [
        rec_weight, 
        [thr_rec, alpha, inp_mask, no_rec_mask], 
        neuron_dyn
    ]
    return net_params