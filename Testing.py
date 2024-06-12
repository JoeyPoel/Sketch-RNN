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
        pi = params[0].float()
        mu1 = params[1][:, :, 0].float()
        mu2 = params[1][:, :, 1].float()
        sigma1 = params[2][:, :, 0].float()
        sigma2 = params[2][:, :, 1].float()
        rho = params[3].float()
        pen = params[4].float()

        # Sample from the output distribution
        next_stroke = sample_from_params(pi, mu1, mu2, sigma1, sigma2, rho, pen)
        strokes.append(next_stroke)

        input_seq = torch.tensor(next_stroke).unsqueeze(0).unsqueeze(0).to(device).float()

    return np.array(strokes)


def sample_from_params(pi, mu1, mu2, sigma1, sigma2, rho, pen):
    """Sample stroke parameters from the given model parameters."""
    pi = F.softmax(pi.detach(), dim=-1).cpu().numpy().squeeze()
    pen = F.softmax(pen.detach(), dim=-1).cpu().numpy().squeeze()

    mu1 = mu1.detach().cpu().numpy().squeeze()
    mu2 = mu2.detach().cpu().numpy().squeeze()
    sigma1 = sigma1.detach().cpu().numpy().squeeze()
    sigma2 = sigma2.detach().cpu().numpy().squeeze()
    rho = rho.detach().cpu().numpy().squeeze()

    idx = np.random.choice(len(pi), p=pi)
    idx = idx % mu1.shape[0]

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
    model = load_model(checkpoint_path, args)
    strokes = sample(model)
    strokes_to_svg(strokes, filename=output_svg)


def parse_args():
    """Parse command-line arguments."""
    parser = hparam_parser()
    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = 'model2_save/checkpoint_epoch_1.pth'

    generate_svg(args, checkpoint_path)
