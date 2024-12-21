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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.data import QM9DataModule
from pytorch_lightning import seed_everything
from src.models import AtomwisePostProcessing
from src.models.painn import PaiNN# this is the working one!
import wandb

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
    parser.add_argument('--num_message_passing_layers', default=6, type=int)
    parser.add_argument('--num_features', default=128, type=int)
    parser.add_argument('--num_outputs', default=1, type=int)
    parser.add_argument('--num_rbf_features', default=20, type=int)
    parser.add_argument('--num_unique_atoms', default=100, type=int)
    parser.add_argument('--cutoff_dist', default=5.0, type=float)

    # Training
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--num_epochs', default=1000, type=int)

    #parser.add_argument('--clip_value', default=1000, type=int)

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
    'method': 'grid',  # Options: 'random', 'grid', 'bayes'
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        #'num_message_passing_layers': {'values': [1]}, # [2, 3, 4, 5]
        #'clip_value': {'values': [1, 10, 100]}
        'lr': {'values': [5e-4, 4]}, # [1e-3, 5e-4, 1e-4]
        #'batch_size_train': {'values': [32, 64, 100]},
        #'weight_decay': {'values': [0.01]} # [0.01, 0.001, 0.0001]
    }
}

sweep_id = wandb.sweep(sweep_config, project="layeropt_6")


def main():
    # Parse static arguments
    args = cli()
    seed_everything(args.seed)

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
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        # Early stopping setup
        smoothed_loss = 0.0
        best_val_loss_smooth = float('inf')
        patience = 30  # Stop training if no improvement after 20 epochs
        patience_counter = 0
        alpha = 0.9
        
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
                #if epoch < 10: torch.nn.utils.clip_grad_norm_(painn.parameters(), max_norm=args.clip_value) 
                #else: 
                torch.nn.utils.clip_grad_value_(painn.parameters(), clip_value=config.get("clip_value", 100))
                optimizer.step()

                train_loss += loss.item()

                # Compute MAE
                mae_step = F.l1_loss(preds, batch.y, reduction='sum').item()
                train_mae += mae_step

            train_loss /= len(dm.data_train)
            train_mae /= len(dm.data_train)  # Normalize MAE by dataset size

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
            
            smoothed_loss = alpha * val_loss + (1 - alpha) * smoothed_loss #  Smooth loss
            scheduler.step(smoothed_loss) # update rlr
            pbar.set_postfix_str(f'Train loss: {train_loss:.3e}, Val loss: {val_loss:.3e}')

            # Log metrics to wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mae': train_mae,
                'val_mae': val_mae,
            })

            # Early stopping
            if (best_val_loss_smooth - smoothed_loss) > 0.0000001:
                best_val_loss_smooth = smoothed_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

# Test evaluation
        mae = 0
        painn.eval()
        with torch.no_grad():
            for batch in dm.test_dataloader():
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
                mae += F.l1_loss(preds, batch.y, reduction='sum')
        mae /= len(dm.data_test)
        unit_conversion = dm.unit_conversion[args.target]
        wandb.log({'Test MAE': unit_conversion(mae.item())})
        #print(os.getcwd() + '/src/results/model_{wandb.run.id}.pth')
        torch.save(painn.state_dict(), f'./src/results/model_{wandb.run.id}.pth')

# Run the sweep agent
wandb.agent(sweep_id, function=main)