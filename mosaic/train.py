import os
import pickle
import time
import matplotlib
import matplotlib.pyplot as plt
from einops import rearrange
from jax import vmap, pmap, jit, value_and_grad, local_device_count
from jax.example_libraries import optimizers
from jax.lax import scan, cond
import jax.numpy as np
import jax.random as random
from utils import calc_mosaic_stats, pruner, add_noise

matplotlib.use('Agg')

def train_mosaic(key, n_batch, n_inp, n_rec, n_out, thr_rec, tau_rec, 
                 lr, lr_dropstep, w_gain, grad_clip, train_dl, test_dl, 
                 model, param_initializer, sparser, decoder, 
                 noise_start_step, prune_start_step, prune_thr, 
                 noise_std, n_epochs, n_core, W_mask, lambda_con,
                 target_fr, lambda_fr, beta, dataset_name):
    
    key, key_model = random.split(key, 2)
    n_devices = local_device_count()

    def net_step(net_params, x_t):
        ''' Single network inference (x_t -> yhat_t)
        '''
        net_params, z_rec = model(net_params, x_t)
        return net_params, z_rec

    @jit
    def predict(weight, X):
        """ Scans over time and return predictions. """
        _, net_const, net_dyn = param_initializer(
            key, n_inp, n_rec, n_out, n_core, thr_rec, tau_rec, w_gain
        )
        _, z_rec = scan(net_step, [weight, net_const, net_dyn], X, length=100)
        z_rec = rearrange(z_rec, 't o -> o t') 
        Yhat = decoder(z_rec[-n_out:]) 
        return Yhat, z_rec

    v_predict = vmap(predict, in_axes=(None, 0))
    p_predict = pmap(v_predict, in_axes=(None, 0))

    def loss(key, weight, X, Y, epoch):
        """ Calculates CE loss after predictions. """
        X = rearrange(X, '(d b) t i -> d b t i', d=n_devices)
        Y = rearrange(Y, '(d b) o -> d b o', d=n_devices)

        weight = cond(
            epoch >= noise_start_step, 
            lambda weight, key : add_noise(weight, key, noise_std),
            lambda weight, key : weight,
            weight, key
        )

        Yhat, z_rec = p_predict(weight, X)
        num_correct = np.sum(np.equal(np.argmax(Yhat, 2), np.argmax(Y, 2)))
        loss_ce = -np.mean(np.sum(Yhat * Y, axis=2, dtype=np.float32))
        loss_sp =  sparser(weight, W_mask)
        loss_fr = np.mean(target_fr - 10 * np.mean(z_rec)) ** 2 
        loss_total = loss_ce + loss_sp * lambda_con + loss_fr * lambda_fr
        loss_values = [num_correct, 10 * np.mean(z_rec), loss_ce, loss_sp, loss_fr]
        return loss_total, loss_values
 
    @jit
    def update(key, epoch, weight, X, Y, opt_state):
        value, grads = value_and_grad(loss, has_aux=True, argnums=(1))(key, weight, X, Y, epoch)
        grads = np.clip(grads, -grad_clip, grad_clip)
        opt_state = opt_update(0, grads, opt_state)
        return get_params(opt_state), opt_state, value

    def one_hot(x, n_class):
        return np.array(x[:, None] == np.arange(n_class), dtype=np.float32)

    def total_correct(weight, X, Y):
        X = rearrange(X, '(d b) t i -> d b t i', d=n_devices)
        Y = rearrange(Y, '(d b) -> d b', d=n_devices)
        Yhat, _ = p_predict(weight, X)
        acc = np.sum(np.equal(np.argmax(Yhat, 2), Y[0]))
        return acc

    pw_lr = optimizers.piecewise_constant([lr_dropstep], [lr, lr/10])
    opt_init, opt_update, get_params = optimizers.adam(step_size=pw_lr)
    weight, _, _ = param_initializer(
        key, n_inp, n_rec, n_out, n_core, thr_rec, tau_rec, w_gain
    )
    opt_state = opt_init(weight)

    # Training loop
    train_loss = []
    train_step = 0
    for epoch in range(n_epochs):
        t = time.time()
        acc = 0
        for batch_idx, (x, y) in enumerate(train_dl):
            y = one_hot(y, n_out)
            key, _ = random.split(key)
            weight, opt_state, (L, [tot_correct, _, _, _, _]) = update(
                key, epoch, weight, x, y, opt_state
            )
            weight = pruner(train_step, prune_start_step, prune_thr, weight)
            train_loss.append(L)
            train_step += 1
            acc += tot_correct
        
        # Training logs
        train_acc = 100*acc/((batch_idx+1)*n_batch)
        elapsed_time = time.time() - t
        print(f'Epoch: [{epoch}/{n_epochs}] - Loss: {L:.2f} - '
              f'Training acc: {train_acc:.2f} - t: {elapsed_time:.2f} sec')
        # Export connectivity matrix
        plt.matshow(np.absolute(weight)>0)
        plt.savefig('connectivity_matrix.png')
        plt.close()
        # occupancy = calc_mosaic_stats(weight, W_mask, beta)
        if epoch % 50 == 0:
            # Save training state
            trained_params = optimizers.unpack_optimizer_state(opt_state)
            checkpoint_path = os.path.join('checkpoints', "checkpoint.pkl")
            with open(checkpoint_path, "wb") as file:
                pickle.dump(trained_params, file)

    # Testing Loop
    if dataset_name == 'shd':
        shd_test_loader = test_dl
    elif dataset_name == 'ssc':
        ssc_test_loader = test_dl
    elif dataset_name == 'all':
        shd_test_loader, ssc_test_loader = test_dl

    # SHD
    acc = 0; test_acc_shd = 0
    if dataset_name in ['shd', 'all']:
        for batch_idx, (x, y) in enumerate(shd_test_loader):
            acc += total_correct(weight, x, y)
        test_acc_shd = 100*acc/((batch_idx+1)*n_batch)
    print(f'SHD Test Accuracy: {test_acc_shd:.2f}')

    # SSC
    acc = 0 ; test_acc_ssc = 0
    if dataset_name in ['ssc', 'all']:
        for batch_idx, (x, y) in enumerate(ssc_test_loader):
            acc += total_correct(weight, x, y)
        test_acc_ssc = 100*acc/((batch_idx+1)*n_batch)
    print(f'SSC Test Accuracy: {test_acc_ssc:.2f}')

    return train_loss, test_acc_shd, test_acc_ssc, weight


if __name__ == '__main__':
    import argparse
    import jax.random as random
    import matplotlib
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import ConcatDataset, WeightedRandomSampler
    from data_utils import get_numpy_datasets, NumpyLoader
    from model import lif_forward
    from utils import param_initializer, sparser, generate_mosaic_mask, fr_decoder

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generation.')
    parser.add_argument('--n_batch', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--n_epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=5e-3, help='Initial learning rate.')
    parser.add_argument('--lr_dropstep', type=int, default=2000, help='Step at which the learning rate will be reduced to lr/10.')
    parser.add_argument('--n_inp', type=int, default=128, help='Number of input neurons.')
    parser.add_argument('--n_rec', type=int, default=2048, help='Number of recurrent neurons.')
    parser.add_argument('--n_out', type=int, default=20, help='Number of output units. Set to 20 for SHD, 35 for SSC, 55 for joint training.')
    parser.add_argument('--thr_rec', type=float, default=1, help='Membrane threshold for recurrent neurons.')
    parser.add_argument('--tau_rec', type=float, default=45e-3, help='Membrane time constant for recurrent neurons.')
    parser.add_argument('--w_gain', type=float, default=0.27, help='Gain for initializing recurrent weights.')
    parser.add_argument('--grad_clip', type=float, default=1e3, help='Maximum allowed gradient norm.')
    parser.add_argument('--lambda_con', type=float, default=0.1, help='Regularization parameter for sparsity.')
    parser.add_argument('--lambda_fr', type=float, default=10, help='Regularization parameter for firing rate.')
    parser.add_argument('--target_fr', type=int, default=5, help='Target firing rate for the recurrent layer.')
    parser.add_argument('--beta', type=float, default=8, help='Scaling factor for the cost of increasing the number of hops.')
    parser.add_argument('--noise_str', type=int, default=10, help='Iteration step at which noise injection starts.')
    parser.add_argument('--prune_str', type=int, default=10, help='Iteration step at which pruning starts.')
    parser.add_argument('--prune_thr', type=float, default=0.0005, help='Weight threshold for pruning.')
    parser.add_argument('--noise_std', type=float, default=0.05, help='Standard deviation of noise added during inference.')
    parser.add_argument('--n_core', type=int, default=64, help='Number of cores in the Mosaic architecture.')
    parser.add_argument('--dataset', type=str, default='shd', help='Dataset to use for training. Options are "shd", "ssc", or "all".')
    args = parser.parse_args()

    train_ds, test_ds = get_numpy_datasets(dataset_name=args.dataset, n_inp=args.n_inp)

    if args.dataset in ['shd','ssc']: 
        train_dl = NumpyLoader(train_ds[0], batch_size=args.n_batch, num_workers=0, 
                               drop_last=True)
        test_dl  = NumpyLoader(test_ds[0], batch_size=args.n_batch, num_workers=0,
                               drop_last=True)

    if args.dataset == 'all':
        shd_train_ds, ssc_train_ds = train_ds
        train_samp_w = torch.cat(((1/len(shd_train_ds)) * torch.ones(len(shd_train_ds)), 
                                  (1/len(ssc_train_ds)) * torch.ones(len(ssc_train_ds))))
        train_sampler = WeightedRandomSampler(train_samp_w, len(train_samp_w))
        train_dl = NumpyLoader(ConcatDataset(train_ds), batch_size=args.n_batch, 
                               num_workers=0, drop_last=True, sampler=train_sampler)
        shd_test_ds, ssc_test_ds = test_ds
        shd_test_dl  = NumpyLoader(shd_test_ds, batch_size=args.n_batch, num_workers=0,
                                   drop_last=True)
        ssc_test_dl  = NumpyLoader(ssc_test_ds, batch_size=args.n_batch, num_workers=0,
                                   drop_last=True)
        test_dl = [shd_test_dl, ssc_test_dl]


    train_loss, test_acc_shd, test_acc_ssc, weights = train_mosaic(
        key=random.PRNGKey(args.seed), 
        n_batch=args.n_batch, 
        n_inp=args.n_inp,
        n_rec=args.n_rec,
        n_out=args.n_out,
        thr_rec=args.thr_rec,
        tau_rec=args.tau_rec,
        lr=args.lr,
        lr_dropstep=args.lr_dropstep,
        w_gain=args.w_gain,
        grad_clip=args.grad_clip,
        train_dl=train_dl,
        test_dl=test_dl,
        model=lif_forward,
        param_initializer=param_initializer,
        sparser=sparser,
        decoder=fr_decoder,
        noise_start_step=args.noise_str, 
        prune_start_step=args.prune_str, 
        prune_thr=args.prune_thr,
        noise_std=args.noise_std,
        n_epochs=args.n_epochs,
        n_core=args.n_core,
        W_mask=generate_mosaic_mask(args.n_rec, args.n_core, args.beta),
        lambda_con=args.lambda_con,
        target_fr=args.target_fr,
        lambda_fr=args.lambda_fr,
        beta=args.beta,
        dataset_name=args.dataset
    )