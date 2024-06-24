import torch
import torch.nn.functional as F
import numpy as np
import svgwrite
import argparse
import warnings

from sketch_rnn import SketchRNN
from sketch_rnn.hparams import hparam_parser


def load_model(checkpoint_path, args):
    """Load the SketchRNN model from the checkpoint."""
    model = SketchRNN(args).to(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def sample(model, z=None, max_seq_len=250):
    """Sample a sequence of strokes from the model."""
    model.eval()
    device = next(model.parameters()).device

    if z is None:
        z = torch.randn(1, model.z_size).to(device).float()

    state = torch.tanh(model.init(z)).chunk(2, dim=-1)
    input_seq = torch.zeros(1, 1, 5).to(device).float()  # Start with a zero input

    strokes = []
    for _ in range(max_seq_len):
        z_rep = z[:, None, :].expand(1, 1, -1)
        input_cat = torch.cat((input_seq, z_rep), dim=-1).float()
        output, state = model.decoder(input_cat, state)
        params = model.param_layer(output)

        # Unpack the params tuple (based on shapes of params)
        pi = params[0].squeeze().float()
        mu1 = params[1].squeeze().float()
        mu2 = params[2].squeeze().float()
        sigma1 = params[3].squeeze().float()
        sigma2 = params[4].squeeze().float()
        rho = params[5].squeeze().float()
        pen = params[6].squeeze().float()

        # Print shapes for debugging
        print("pi shape:", pi.shape)
        print("mu1 shape:", mu1.shape)
        print("mu2 shape:", mu2.shape)
        print("sigma1 shape:", sigma1.shape)
        print("sigma2 shape:", sigma2.shape)
        print("rho shape:", rho.shape)
        print("pen shape:", pen.shape)

        # Sample from the output distribution
        next_stroke = sample_from_params(pi, mu1, mu2, sigma1, sigma2, rho, pen, model.num_mixture)
        strokes.append(next_stroke)

        input_seq = torch.tensor(next_stroke).unsqueeze(0).unsqueeze(0).to(device).float()

    return np.array(strokes)


def sample_from_params(pi, mu1, mu2, sigma1, sigma2, rho, pen, num_mixture):
    """Sample stroke parameters from the given model parameters."""
    pi = F.softmax(pi.detach(), dim=-1).cpu().numpy()
    pen = F.softmax(pen.detach(), dim=-1).cpu().numpy()

    mu1 = mu1.detach().cpu().numpy()
    mu2 = mu2.detach().cpu().numpy()
    sigma1 = sigma1.detach().cpu().numpy()
    sigma2 = sigma2.detach().cpu().numpy()
    rho = rho.detach().cpu().numpy()

    # Ensure the shapes are correct
    if len(sigma1.shape) == 0: sigma1 = np.array([sigma1])
    if len(sigma2.shape) == 0: sigma2 = np.array([sigma2])
    if len(mu1.shape) == 0: mu1 = np.array([mu1])
    if len(mu2.shape) == 0: mu2 = np.array([mu2])

    # Print parameter shapes for debugging
    print("pi:", pi)
    print("mu1:", mu1)
    print("mu2:", mu2)
    print("sigma1:", sigma1)
    print("sigma2:", sigma2)
    print("rho:", rho)
    print("pen:", pen)

    idx = np.random.choice(len(pi), p=pi)
    if idx >= len(mu1):  # Ensure index is within bounds
        idx = len(mu1) - 1

    mean = [mu1[idx], mu2[idx]]
    cov = [[sigma1[idx] ** 2, rho[idx] * sigma1[idx] * sigma2[idx]],
           [rho[idx] * sigma1[idx] * sigma2[idx], sigma2[idx] ** 2]]
    xy = np.random.multivariate_normal(mean, cov)

    pen_state = np.random.choice(len(pen), p=pen)

    return [xy[0], xy[1], pen_state == 0, pen_state == 1, pen_state == 2]


def strokes_to_svg(strokes, filename='sample.svg'):
    """Convert a sequence of strokes to an SVG file."""
    width, height = 400, 400
    dwg = svgwrite.Drawing(filename, profile='tiny', size=(width, height), viewBox="0 0 400 400")

    scale = 10
    x, y = 200, 200

    lift_pen = 1
    path = dwg.path(d=f"M{x},{y} ", stroke='black', fill='none')

    for dx, dy, p1, p2, p3 in strokes:
        dx *= scale
        dy *= scale

        if lift_pen == 1:
            if path:
                dwg.add(path)
            path = dwg.path(d=f"M{x},{y} ", stroke='black', fill='none')

        x += dx
        y += dy
        path.push(f"L{x},{y}")
        lift_pen = p3

    if path:
        dwg.add(path)

    dwg.save()


def generate_svg(args, checkpoint_path, output_svg='sample1.svg'):
    """Generate an SVG file from a model checkpoint."""
    try:
        model = load_model(checkpoint_path, args)
        strokes = sample(model)
        strokes_to_svg(strokes, filename=output_svg)
    except Exception as e:
        print(f"Error in generate_svg: {e}")


def parse_args():
    """Parse command-line arguments."""
    parser = hparam_parser()
    args, unknown = parser.parse_known_args()

    # Adjust hyperparameters to match the checkpoint
    args.enc_rnn_size = 256  # Correcting to match checkpoint
    args.dec_rnn_size = 512  # Correcting to match checkpoint
    args.z_size = 128
    args.num_mixture = 2  # Match the actual number of components in mu1 and mu2
    
    return args


if __name__ == "__main__":
    args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = 'model12_save/checkpoint_epoch_17.pth'

    generate_svg(args, checkpoint_path)
