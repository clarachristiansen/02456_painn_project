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
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.swa_utils import AveragedModel
from torch.optim.swa_utils import update_bn

class CyclicSWALR(_LRScheduler):
    def __init__(self, optimizer, cycle_length, swa_lr, scale_factor=10, last_epoch=-1):
        """
        Custom SWALR scheduler with periodic reset of the learning rate.
        
        Args:
        - optimizer: Optimizer for which to schedule the learning rate.
        - cycle_length: Number of epochs for each cosine cycle.
        - swa_lr: The final learning rate for SWA.
        - scale_factor: Scaling factor for the initial learning rate.
        - last_epoch: The index of the last epoch when resuming training.
        """
        self.cycle_length = cycle_length
        self.swa_lr = swa_lr
        self.scale_factor = scale_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the learning rate for the current epoch."""
        # Determine the position within the current cycle
        cycle_position = self.last_epoch % self.cycle_length
        lrs = []
        for base_lr in self.base_lrs:
            if cycle_position == 0:  # Reset to the scaled initial LR at the start of a cycle
                lrs.append(base_lr * self.scale_factor)
            else:
                # Linearly anneal within the current cycle
                scaled_initial_lr = base_lr * self.scale_factor
                slope = (scaled_initial_lr - self.swa_lr) / (self.cycle_length - 1)
                lrs.append(scaled_initial_lr - cycle_position * slope)
                #lrs.append(scaled_initial_lr - scaled_initial_lr * self.swa_factor * cycle_position)
        return lrs




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
    parser.add_argument('--num_message_passing_layers', default=5, type=int)
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
        #'lr': {'values': [5e-4, 4]}, # [1e-3, 5e-4, 1e-4]
        #'batch_size_train': {'values': [32, 64, 100]},
        #'weight_decay': {'values': [0.01]} # [0.01, 0.001, 0.0001]
        'swa_lr': {'values': [1e-14, 1e-15, 1e-16]},
        'cycle_length_swa': {'values': [20, 50, 100]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="swa_5")


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
        painn.load_state_dict(torch.load('./src/results/model_dvyocn32.pth', map_location=torch.device(device)))
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

        #SWA
        swa_model = AveragedModel(painn)
        swa_scheduler = CyclicSWALR(optimizer, cycle_length=config.get("cycle_length_swa", 0.05), swa_lr=config.get("swa_lr", 0.05), scale_factor=1e-8)
        args.num_epochs = int(config.get("cycle_length_swa", args.num_epochs) * 5)


        # Early stopping setup
        
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

            swa_model.update_parameters(painn)
            swa_scheduler.step()
            current_lr = swa_scheduler.get_last_lr()
            wandb.log({'lr': current_lr[0]})


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
        torch.save(swa_model.module.state_dict(), f'./src/results/real_swa_model_{wandb.run.id}.pth')

# Run the sweep agent
wandb.agent(sweep_id, function=main)