"""
PyTorch Model Implementation
Advanced neural network architectures with production best practices
Demonstrates 3+ years of experience with industry ML frameworks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from loguru import logger
import mlflow
from tqdm import tqdm


class TabularDataset(Dataset):
    """Custom Dataset for tabular data."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class AttentionLayer(nn.Module):
    """Self-attention mechanism for feature importance."""

    def __init__(self, input_dim: int, attention_dim: int = 64):
        super().__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.scale = np.sqrt(attention_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended = torch.matmul(attention_weights, V)

        return attended, attention_weights


class ResidualBlock(nn.Module):
    """Residual connection block for deep networks."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return F.relu(x + residual)


class AdvancedNeuralNetwork(nn.Module):
    """
    Advanced neural network with attention, residual connections, and regularization.
    Production-ready architecture for various ML tasks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout: float = 0.3,
        use_attention: bool = True,
        use_residual: bool = True,
        use_batch_norm: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        self.use_residual = use_residual

        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_norm = nn.BatchNorm1d(hidden_dims[0]) if use_batch_norm else nn.Identity()

        # Attention mechanism
        if use_attention:
            self.attention = AttentionLayer(hidden_dims[0])

        # Hidden layers with optional residual connections
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            if use_residual and hidden_dims[i] == hidden_dims[i + 1]:
                self.hidden_layers.append(ResidualBlock(hidden_dims[i], dropout))
            else:
                layer = nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]) if use_batch_norm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                self.hidden_layers.append(layer)

        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional attention weights return."""
        # Input processing
        x = self.input_layer(x)
        x = self.input_norm(x)
        x = F.relu(x)

        # Attention
        attention_weights = None
        if self.use_attention:
            x, attention_weights = self.attention(x.unsqueeze(1))
            x = x.squeeze(1)

        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)

        # Output
        output = self.output_layer(x)

        if return_attention and attention_weights is not None:
            return output, attention_weights
        return output


class PyTorchModelTrainer:
    """
    Production-ready PyTorch model trainer with MLflow integration,
    early stopping, checkpointing, and comprehensive logging.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            device: Device to use (cuda/cpu)
            config: Training configuration
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.config = config or {}

        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.best_loss = float('inf')
        self.patience_counter = 0

        logger.info(f"Initialized trainer on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def setup_training(
        self,
        optimizer_name: str = 'adamw',
        learning_rate: float = 0.001,
        scheduler_name: str = 'plateau',
        loss_fn: str = 'mse'
    ):
        """Setup optimizer, scheduler, and loss function."""
        # Optimizer
        if optimizer_name.lower() == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=0.01
            )
        elif optimizer_name.lower() == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                nesterov=True
            )

        # Scheduler
        if scheduler_name.lower() == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_name.lower() == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=50,
                eta_min=1e-6
            )

        # Loss function
        if loss_fn.lower() == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_fn.lower() == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_fn.lower() == 'crossentropy':
            self.criterion = nn.CrossEntropyLoss()

        logger.info(f"Training setup: {optimizer_name}, {scheduler_name}, {loss_fn}")

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_features, batch_labels in tqdm(dataloader, desc="Training"):
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_features)
            loss = self.criterion(outputs, batch_labels)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_features, batch_labels in dataloader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)

                total_loss += loss.item()
                all_predictions.append(outputs.cpu().numpy())
                all_labels.append(batch_labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)

        # Calculate metrics
        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)

        metrics = self._calculate_metrics(predictions, labels)
        metrics['loss'] = avg_loss

        return avg_loss, metrics

    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        metrics = {
            'mse': mean_squared_error(labels, predictions),
            'mae': mean_absolute_error(labels, predictions),
            'rmse': np.sqrt(mean_squared_error(labels, predictions)),
            'r2': r2_score(labels, predictions)
        }

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        checkpoint_path: str = 'models/checkpoint.pt',
        use_mlflow: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train model with early stopping and checkpointing.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            checkpoint_path: Path to save best model
            use_mlflow: Whether to log to MLflow

        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }

        if use_mlflow:
            mlflow.start_run()
            mlflow.log_params({
                'model': self.model.__class__.__name__,
                'optimizer': self.optimizer.__class__.__name__,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'epochs': epochs,
            })

        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)

            # Validation
            val_loss, val_metrics = self.validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)

            # Logging
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val RMSE: {val_metrics['rmse']:.4f}"
            )

            if use_mlflow:
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                }, step=epoch)

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early stopping and checkpointing
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(checkpoint_path)
                logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        if use_mlflow:
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.end_run()

        return history

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        logger.info(f"Loaded checkpoint from {path}")


def create_model(input_dim: int, output_dim: int, config: Dict[str, Any]) -> nn.Module:
    """Factory function to create model from config."""
    model = AdvancedNeuralNetwork(
        input_dim=input_dim,
        hidden_dims=config.get('hidden_dims', [256, 128, 64]),
        output_dim=output_dim,
        dropout=config.get('dropout', 0.3),
        use_attention=config.get('use_attention', True),
        use_residual=config.get('use_residual', True),
        use_batch_norm=config.get('use_batch_norm', True)
    )
    return model


if __name__ == "__main__":
    # Example usage
    input_dim = 20
    output_dim = 1

    # Create model
    config = {
        'hidden_dims': [256, 128, 64],
        'dropout': 0.2,
        'use_attention': True,
        'use_residual': True
    }
    model = create_model(input_dim, output_dim, config)

    # Create dummy data
    X_train = np.random.randn(1000, input_dim)
    y_train = np.random.randn(1000, output_dim)
    X_val = np.random.randn(200, input_dim)
    y_val = np.random.randn(200, output_dim)

    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Train model
    trainer = PyTorchModelTrainer(model, config=config)
    trainer.setup_training(optimizer_name='adamw', learning_rate=0.001)
    history = trainer.fit(train_loader, val_loader, epochs=10, use_mlflow=False)

    logger.info("Training completed successfully")
