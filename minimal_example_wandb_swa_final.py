"""
Basic example of how to train the PaiNN model to predict the QM9 property
"internal energy at 0K". This property (and the majority of the other QM9
properties) is computed as a sum of atomic contributions.
"""
import torch
import argparse
import os
import pandas as pd
import time
from tqdm import trange
import torch.nn.functional as F
from src.data import QM9DataModule
from pytorch_lightning import seed_everything
from src.models import AtomwisePostProcessing
from src.models.painn import PaiNN # this is the working one!
import wandb
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.swa_utils import update_bn

print('currrent working directory: ', os.getcwd())
load = True
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0)

    # Data
    parser.add_argument('--target', default=7, type=int) # 7 => Internal energy at 0K
    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--batch_size_train', default=100, type=int)
    parser.add_argument('--batch_size_inference', default=1000, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--splits', nargs=3, default=[110000, 10000, 10831], type=int) # [num_train, num_val, num_test]
    parser.add_argument('--subset_size', default=None, type=int)

    # Model
    parser.add_argument('--num_message_passing_layers', default=3, type=int)
    parser.add_argument('--num_features', default=128, type=int)
    parser.add_argument('--num_outputs', default=1, type=int)
    parser.add_argument('--num_rbf_features', default=20, type=int)
    parser.add_argument('--num_unique_atoms', default=100, type=int)
    parser.add_argument('--cutoff_dist', default=5.0, type=float)

    # Training
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--num_epochs', default=200, type=int)

    args = parser.parse_args()
    return args

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

device = get_device()
print(device)

wandb.login(key='eff5a31d6dfda82af022ae7c5286724a57c42f8c')
# Initialize wandb sweep config
sweep_config = {
    'method': 'bayes',  # Options: 'random', 'grid', 'bayes'
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'lr_swa': {'values': [1e-7, 1e-8, 1e-9]},
        'cycle_length_swa': {'values': [10, 20, 50]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="QM9_PaiNN_swa_final")


def main():
    # Parse static arguments
    args = cli()

    # wandb configuration (dynamic hyperparameters)
    with wandb.init(config=args.__dict__) as run:
        config = wandb.config  # Access wandb-specified parameters
        print(wandb.run.id)

        # Use a mix of CLI and wandb settings
        dm = QM9DataModule(
            target=args.target,
            data_dir=args.data_dir,
            batch_size_train=config.get("batch_size_train", args.batch_size_train),
            batch_size_inference=args.batch_size_inference,
            num_workers=args.num_workers,
            splits=args.splits,
            seed=args.seed,
            subset_size=args.subset_size,
        )
        dm.prepare_data()
        dm.setup()
        y_mean, y_std, atom_refs = dm.get_target_stats(
        remove_atom_refs=True, divide_by_atoms=True
        )
        painn = PaiNN(
            num_message_passing_layers=config.get("num_message_passing_layers", args.num_message_passing_layers),
            num_features=args.num_features,
            num_outputs=args.num_outputs,
            num_rbf_features=args.num_rbf_features,
            num_unique_atoms=args.num_unique_atoms,
            cutoff_dist=args.cutoff_dist,
            device=device,
        )
        painn.load_state_dict(torch.load('./src/results/model_hdzfifii.pth'))
        post_processing = AtomwisePostProcessing(
        args.num_outputs, y_mean, y_std, atom_refs
        )
        painn.to(device)
        post_processing.to(device)

        # Optimizer hyperparameters come from wandb
        optimizer = torch.optim.AdamW(
            painn.parameters(),
            lr=config.get("lr", args.lr),
            weight_decay=config.get("weight_decay", args.weight_decay),
        )
        swa_model = AveragedModel(painn)
        swa_scheduler = SWALR(optimizer, swa_lr=config.get("lr_swa", 0.05), anneal_epochs=config.get("cycle_length_swa", 10), anneal_strategy='linear')

        start_swa = True

        pbar = trange(args.num_epochs)
        for epoch in pbar:
            painn.train()
            train_loss = 0.0
            train_mae = 0.0  # Initialize train MAE accumulator

            for batch in dm.train_dataloader():
                batch = batch.to(device)
                atomic_contributions = painn(
                    atoms=batch.z,
                    atom_positions=batch.pos,
                    graph_indexes=batch.batch
                )
                preds = post_processing(
                    atoms=batch.z,
                    graph_indexes=batch.batch,
                    atomic_contributions=atomic_contributions,
                )
                loss = F.mse_loss(preds, batch.y, reduction='sum') / len(batch.y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Compute MAE
                mae_step = F.l1_loss(preds, batch.y, reduction='sum').item()
                train_mae += mae_step

            train_loss /= len(dm.data_train)
            train_mae /= len(dm.data_train)  # Normalize MAE by dataset size

            # SWA update
            if start_swa:# * 0.75:  # Start SWA in the last 25% epochs
                swa_model.update_parameters(painn)
                swa_scheduler.step(epoch=epoch)


            # Validation Loop
            painn.eval()
            val_loss = 0.0
            val_mae = 0.0  # Initialize val MAE accumulator
            with torch.no_grad():
                for batch in dm.val_dataloader():
                    batch = batch.to(device)
                    atomic_contributions = painn(
                        atoms=batch.z,
                        atom_positions=batch.pos,
                        graph_indexes=batch.batch,
                    )
                    preds = post_processing(
                        atoms=batch.z,
                        graph_indexes=batch.batch,
                        atomic_contributions=atomic_contributions,
                    )
                    val_loss += F.mse_loss(preds, batch.y, reduction='sum').item()
                    
                    # Compute MAE
                    mae_step = F.l1_loss(preds, batch.y, reduction='sum').item()
                    val_mae += mae_step

            val_loss /= len(dm.data_val)
            val_mae /= len(dm.data_val)  # Normalize MAE by dataset size
            pbar.set_postfix_str(f'Train loss: {train_loss:.3e}, Val loss: {val_loss:.3e}')

            # Log metrics to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mae': train_mae,
                'val_mae': val_mae,
            })

# Test evaluation
        update_bn(dm.train_dataloader(), swa_model)
        mae = 0
        painn.eval()
        with torch.no_grad():
            for batch in dm.test_dataloader():
                batch = batch.to(device)
                atomic_contributions = swa_model(
                    atoms=batch.z,
                    atom_positions=batch.pos,
                    graph_indexes=batch.batch,
                )
                preds = post_processing(
                    atoms=batch.z,
                    graph_indexes=batch.batch,
                    atomic_contributions=atomic_contributions,
                )
                mae += F.l1_loss(preds, batch.y, reduction='sum')
        mae /= len(dm.data_test)
        wandb.log({'Test MAE': mae.item()})
        torch.save(swa_model.module.state_dict(), f'./src/results/swa_model_{wandb.run.id}.pth')

# Run the sweep agent
wandb.agent(sweep_id, function=main)
