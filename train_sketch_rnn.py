import os
import argparse
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
import torch.nn.functional as F

from sketch_rnn.hparams import hparam_parser
from sketch_rnn.utils import AverageMeter
from sketch_rnn.dataset import load_sketches, SketchRNNDataset, collate_drawings
from sketch_rnn.model import SketchRNN, model_step
from sketch_rnn.checkpoint import ModelCheckpoint


def collate_drawings_fn(batch, max_len):
    # Pad sequences to max_len and return correct lengths
    batch = [torch.tensor(d, dtype=torch.float32) for d in batch]
    lengths = [min(len(d), max_len) for d in batch]
    padded_batch = torch.zeros(len(batch), max_len, 2)
    for i, d in enumerate(batch):
        end = lengths[i]
        padded_batch[i, :end] = d[:end]
    return padded_batch, torch.tensor(lengths, dtype=torch.long)


def compute_loss(params, z_mean, z_logvar, targets):
    # Select the predicted values
    predicted_values = params[0][:, :targets.shape[1], :2]  # Slice to match target length

    # Compute reconstruction loss
    reconstruction_loss = F.mse_loss(predicted_values, targets)

    # Compute KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())

    # Combine the two losses
    loss = reconstruction_loss + kl_loss

    return loss


def train_epoch(model, train_loader, optimizer, scheduler, device, grad_clip=None):
    model.train()
    epoch_loss = 0.0
    with tqdm(total=len(train_loader.dataset)) as progress_bar:
        for i, batch in enumerate(train_loader):
            batch_input, batch_lengths = batch
            batch_input = batch_input.to(device)
            batch_lengths = batch_lengths.to(device)

            optimizer.zero_grad()
            output = model(batch_input, batch_lengths)
            if output is None:
                continue  # Skip the batch if the output is None (due to data issues)

            # Unpack the output tuple
            params, z_mean, z_logvar = output

            # Extract targets and adjust the length to match predicted values
            targets = batch_input[:, 1:params[0].shape[1] + 1, :2]

            # Compute the loss
            loss = compute_loss(params, z_mean, z_logvar, targets)

            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(loss=epoch_loss / (i + 1))
            progress_bar.update(batch_input.size(0))

    if scheduler is not None:
        scheduler.step()

    return epoch_loss / len(train_loader)



@torch.no_grad()
def eval_epoch(model, data_loader, device):
    model.eval()
    loss_meter = AverageMeter()
    with tqdm(total=len(data_loader.dataset)) as progress_bar:
        for i, (data, lengths) in enumerate(data_loader):
            data = data.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            # Use the modified model_step function to compute the loss
            loss = model_step(model, data, lengths)
            if loss is None:  # Skip this batch if loss is None
                print(f'Batch {i} - loss is None')
                continue

            loss_meter.update(loss.item(), data.size(0))
            progress_bar.set_postfix(loss=loss_meter.avg)
            progress_bar.update(data.size(0))

    return loss_meter.avg


def train_sketch_rnn(args):
    torch.manual_seed(884)
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda') if use_gpu else torch.device('cpu')
    saver = ModelCheckpoint(args.save_dir) if args.save_dir is not None else None

    # Initialize train and val datasets
    train_sketches, valid_sketches, test_sketches = load_sketches(args.data_dir, args)
    print(f"Number of training sketches: {len(train_sketches)}")
    print(f"Number of validation sketches: {len(valid_sketches)}")
    print(f"Number of test sketches: {len(test_sketches)}")
    train_data = SketchRNNDataset(
        train_sketches,
        max_len=args.max_seq_len,
        random_scale_factor=args.random_scale_factor,
        augment_stroke_prob=args.augment_stroke_prob
    )
    val_data = SketchRNNDataset(
        valid_sketches,
        max_len=args.max_seq_len,
        scale_factor=train_data.scale_factor,
        random_scale_factor=args.random_scale_factor,
        augment_stroke_prob=args.augment_stroke_prob
    )


    # Initialize data loaders
    collate_fn = partial(collate_drawings_fn, max_len=args.max_seq_len)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=use_gpu,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=use_gpu,
        num_workers=args.num_workers
    )

    # Model & optimizer
    model = SketchRNN(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    # Load checkpoint if available
    start_epoch = 0
    if saver is not None:
        start_epoch, model, optimizer = saver.load(model, optimizer)
        if start_epoch is None:
            start_epoch = 0

    for epoch in range(start_epoch, args.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, args.grad_clip)
        val_loss = eval_epoch(model, val_loader, device)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {val_loss:.4f}')
        if saver is not None:
            saver.save(epoch, model, optimizer, train_loss, val_loss)
        time.sleep(0.5)  # Avoids progress bar issue


if __name__ == '__main__':
    hp_parser = hparam_parser()
    parser = argparse.ArgumentParser(parents=[hp_parser])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default= 20)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    train_sketch_rnn(args)
