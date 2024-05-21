import time
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial

from sketch_rnn.hparams import hparam_parser
from sketch_rnn.utils import AverageMeter
from sketch_rnn.dataset import SketchRNNDataset, load_strokes, collate_drawings
from sketch_rnn.model import SketchRNN, model_step
from sketch_rnn.checkpoint import ModelCheckpoint

def collate_drawings_fn(x, max_len):
    return collate_drawings(x, max_len)

def train_epoch(model, data_loader, optimizer, scheduler, device, grad_clip=None):
    model.train()
    loss_meter = AverageMeter()
    with tqdm(total=len(data_loader.dataset)) as progress_bar:
        for data, lengths in data_loader:
            data = data.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            # Training step
            optimizer.zero_grad()
            loss = model_step(model, data, lengths)
            if loss is None:  # Skip this batch if loss is None
                continue
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            # Update loss meter and progress bar
            loss_meter.update(loss.item(), data.size(0))
            progress_bar.set_postfix(loss=loss_meter.avg)
            progress_bar.update(data.size(0))

    return loss_meter.avg


@torch.no_grad()
def eval_epoch(model, data_loader, device):
    model.eval()
    loss_meter = AverageMeter()
    with tqdm(total=len(data_loader.dataset)) as progress_bar:
        for data, lengths in data_loader:
            data = data.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            loss = model_step(model, data, lengths)
            if loss is None:  # Skip this batch if loss is None
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
    train_strokes, valid_strokes, test_strokes = load_strokes(args.data_dir, args)
    train_data = SketchRNNDataset(
        train_strokes,
        max_len=args.max_seq_len,
        random_scale_factor=args.random_scale_factor,
        augment_stroke_prob=args.augment_stroke_prob
    )
    val_data = SketchRNNDataset(
        valid_strokes,
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
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    train_sketch_rnn(args)
