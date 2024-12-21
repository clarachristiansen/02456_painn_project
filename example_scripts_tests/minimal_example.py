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
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--num_epochs', default=5, type=int)

    args = parser.parse_args()
    return args

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def main():
    args = cli()
    seed_everything(args.seed)
    device = get_device()
    print(device)

    dm = QM9DataModule(
        target=args.target,
        data_dir=args.data_dir,
        batch_size_train=args.batch_size_train,
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
        num_message_passing_layers=args.num_message_passing_layers,
        num_features=args.num_features,
        num_outputs=args.num_outputs, 
        num_rbf_features=args.num_rbf_features,
        num_unique_atoms=args.num_unique_atoms,
        cutoff_dist=args.cutoff_dist,
        device=device
    )
    if load:
        painn.load_state_dict(torch.load('./src/results/model_v1.pth', weights_only=True))
        print('Painn model loaded!')

    post_processing = AtomwisePostProcessing(
        args.num_outputs, y_mean, y_std, atom_refs
    )

    painn.to(device)
    post_processing.to(device)

    optimizer = torch.optim.AdamW(
        painn.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Early stopping setup
    best_val_loss = float('inf')
    patience = 20  # Stop training if no improvement after 20 epochs
    patience_counter = 0

    # Save losses
    train_loss_list = []
    val_loss_list = []

    # Train loss calculation
    painn.train()
    pbar = trange(args.num_epochs)
    for epoch in pbar:
        start_time = time.time()
        loss_epoch = 0.
        for batch_idx, batch in enumerate(dm.train_dataloader()):
            #print(len(batch))
            
            batch = batch.to(device)
            #print('LABEL:', batch.y)

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
            #print(preds)
            
            loss_step = F.mse_loss(preds, batch.y, reduction='sum')

            loss = loss_step / len(batch.y)
            #print(f"Batch {batch_idx} Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(painn.parameters(), max_norm=1.0) 
            optimizer.step()

            loss_epoch += loss_step.detach().item()
        loss_epoch /= len(dm.data_train)
        train_loss_list.append(loss_epoch)
        #pbar.set_postfix_str(f'Train loss: {loss_epoch:.3e}')

        # Validation loss calculation
        painn.eval()
        val_loss = 0.
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
        val_loss /= len(dm.data_val)
        val_loss_list.append(val_loss)
        pbar.set_postfix_str(f'Train loss: {loss_epoch:.3e}, Val loss: {val_loss:.3e}')

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.3e}")
                break
        
        dict = {'train_loss': train_loss_list, 'val_loss': val_loss_list}
        df = pd.DataFrame(dict) 
        df.to_csv('./src/results/results_v1.csv')
        print("--- %s seconds ---" % (time.time() - start_time))

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
    print(f'Test MAE: {unit_conversion(mae):.3f}')
    torch.save(painn.state_dict(), './src/results/model_v1.pth')



if __name__ == '__main__':
    main()