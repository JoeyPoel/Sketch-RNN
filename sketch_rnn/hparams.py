import warnings
import argparse

def hparam_parser():
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_argument_group('hyperparameters')

    # architecture params
    group.add_argument('--max_seq_len', type=int, default=250)
    group.add_argument('--enc_model', type=str, default='lstm')
    group.add_argument('--dec_model', type=str, default='layer_norm')
    group.add_argument('--enc_rnn_size', type=int, default=256) 
    group.add_argument('--dec_rnn_size', type=int, default=256)  # Reduced to match enc_rnn_size for balance
    group.add_argument('--z_size', type=int, default=128)
    group.add_argument('--num_mixture', type=int, default=20)
    group.add_argument('--r_dropout', type=float, default=0.1)  # Dropout rate reduced to mitigate overfitting risk
    # group.add_argument('--input_dropout', type=float, default=0.0) # Not recommended
    # group.add_argument('--output_dropout', type=float, default=0.0) # Not recommended

    # loss params
    group.add_argument('--kl_weight', type=float, default=0.5)
    group.add_argument('--kl_weight_start', type=float, default=0.01)  # Lower starting KL weight to encourage early exploration
    group.add_argument('--kl_tolerance', type=float, default=0.2)
    group.add_argument('--kl_decay_rate', type=float, default=0.99995)  # Gradual decay rate to stabilize training
    group.add_argument('--reg_covar', type=float, default=1e-5)  # Slightly reduced regularization for covariance

    # training params
    group.add_argument('--batch_size', type=int, default=64)  # Reduced batch size to possibly improve gradient updates
    group.add_argument('--lr', type=float, default=0.001)
    group.add_argument('--lr_decay', type=float, default=0.9999)  # Moderate decay rate to adapt learning rate
    group.add_argument('--min_lr', type=float, default=0.0001)  # Set a minimum learning rate for stability
    group.add_argument('--grad_clip', type=float, default=5.0)  # Increased grad clip to manage larger gradients

    # dataset & data augmentation params
    group.add_argument('--data_set', type=str, default='cat.npz')
    group.add_argument('--random_scale_factor', type=float, default=0.10)  # Slightly reduced random scale factor
    group.add_argument('--augment_stroke_prob', type=float, default=0.15)  # Increased augmentation probability

    return parser


def hparams(**kwargs):
    parser = hparam_parser()
    hps = parser.parse_args([])
    for key, val in kwargs.items():
        if not hasattr(hps, key):
            warnings.warn("A non-standard hyperparam '%s' was specified." % key)
        setattr(hps, key, val)
    return hps