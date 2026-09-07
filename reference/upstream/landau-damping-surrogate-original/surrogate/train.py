import time
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_and_preprocess
from model import LandauSurrogate
import os


def train(lambda_p=0.001, epochs=150):
    # 1. Load data
    inputs, targets = load_and_preprocess()
    
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)
    print(f"Batches per epoch: {len(loader)} (batch_size=4096)")

    model = LandauSurrogate()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # epochs: training length; lambda_p: physics-penalty strength
    
    print(f"Starting Physics-Informed training (lambda_p={lambda_p})...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        t_epoch = time.perf_counter()

        for batch_x, batch_y in loader:
            # --- Enable gradients w.r.t. inputs for physics penalty ---
            batch_x.requires_grad = True

            optimizer.zero_grad()
            outputs = model(batch_x)

            # 2a. Standard MSE loss
            loss_mse = criterion(outputs, batch_y)

            # 2b. Physics loss: penalize dE/dt > 0 (energy should not grow with time)
            # Derivative of each output element w.r.t. batch_x
            grad_outputs = torch.ones_like(outputs)
            gradients = torch.autograd.grad(
                outputs=outputs,
                inputs=batch_x,
                grad_outputs=grad_outputs,
                create_graph=True,  # Needed for second derivatives through the loss
                retain_graph=True,
            )[0]

            # batch_x[:, 2] is normalized time t
            # Penalize positive slope (unphysical energy increase)
            d_energy_dt = gradients[:, 2]
            loss_physics = torch.mean(torch.relu(d_energy_dt))  # relu keeps only dE/dt > 0

            # 2c. Total loss
            loss = loss_mse + lambda_p * loss_physics
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        epoch_s = time.perf_counter() - t_epoch
        if epoch == 0:
            print(f"Epoch 1 done in {epoch_s:.1f}s (if this is huge, reduce data or max_points_per_file).")
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.6f} "
                f"(MSE: {loss_mse.item():.6f}, Phys: {loss_physics.item():.6f}), "
                f"last epoch {epoch_s:.1f}s"
            )

    # Save checkpoint
    os.makedirs("surrogate/models", exist_ok=True)
    torch.save(model.state_dict(), "surrogate/models/landau_model.pth")
    print("Training complete. Physics-Informed Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_p", type=float, default=0.001, help="Physics loss weight")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    args = parser.parse_args()
    train(lambda_p=args.lambda_p, epochs=args.epochs)