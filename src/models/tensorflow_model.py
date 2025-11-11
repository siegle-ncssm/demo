"""
TensorFlow/Keras Model Implementation
Production-ready deep learning with TF ecosystem
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam, SGD
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from loguru import logger
import mlflow
import mlflow.tensorflow


class TransformerBlock(layers.Layer):
    """Transformer block for sequence or tabular data."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class WideAndDeepModel:
    """
    Wide & Deep architecture for combining memorization and generalization.
    Ideal for recommendation systems and tabular data.
    """

    def __init__(
        self,
        wide_dim: int,
        deep_dim: int,
        deep_hidden_units: List[int],
        output_dim: int,
        dropout_rate: float = 0.3
    ):
        self.wide_dim = wide_dim
        self.deep_dim = deep_dim
        self.deep_hidden_units = deep_hidden_units
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.model = self._build_model()

    def _build_model(self) -> Model:
        """Build wide & deep architecture."""
        # Wide input (linear features)
        wide_input = layers.Input(shape=(self.wide_dim,), name='wide_input')

        # Deep input (dense features)
        deep_input = layers.Input(shape=(self.deep_dim,), name='deep_input')

        # Deep component
        deep = deep_input
        for units in self.deep_hidden_units:
            deep = layers.Dense(units, activation='relu')(deep)
            deep = layers.BatchNormalization()(deep)
            deep = layers.Dropout(self.dropout_rate)(deep)

        # Concatenate wide and deep
        combined = layers.concatenate([wide_input, deep])

        # Output layer
        if self.output_dim == 1:
            output = layers.Dense(1, activation='sigmoid', name='output')(combined)
        else:
            output = layers.Dense(self.output_dim, activation='softmax', name='output')(combined)

        model = Model(inputs=[wide_input, deep_input], outputs=output)
        return model

    def compile_model(
        self,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss: str = 'binary_crossentropy',
        metrics: List[str] = ['accuracy']
    ):
        """Compile model with optimizer and loss."""
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
        else:
            opt = optimizer

        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
        logger.info(f"Model compiled with {optimizer} optimizer")


class AdvancedKerasModel:
    """
    Advanced Keras model with modern architectures and best practices.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.history = None

    def build_residual_network(
        self,
        input_dim: int,
        output_dim: int,
        num_blocks: int = 3,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3
    ) -> Model:
        """Build ResNet-style architecture for tabular data."""
        inputs = layers.Input(shape=(input_dim,))

        # Initial dense layer
        x = layers.Dense(hidden_dim, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)

        # Residual blocks
        for _ in range(num_blocks):
            residual = x
            x = layers.Dense(hidden_dim, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Dense(hidden_dim)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Add()([x, residual])
            x = layers.Activation('relu')(x)

        # Output layer
        if output_dim == 1:
            outputs = layers.Dense(1)(x)
        else:
            outputs = layers.Dense(output_dim, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def build_attention_network(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        ff_dim: int = 512,
        num_transformer_blocks: int = 2,
        dropout_rate: float = 0.1
    ) -> Model:
        """Build attention-based network."""
        inputs = layers.Input(shape=(input_dim,))

        # Embedding layer
        x = layers.Dense(embed_dim)(inputs)
        x = layers.Reshape((1, embed_dim))(x)

        # Transformer blocks
        for _ in range(num_transformer_blocks):
            transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            x = transformer_block(x)

        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(dropout_rate)(x)

        # Output
        if output_dim == 1:
            outputs = layers.Dense(1)(x)
        else:
            outputs = layers.Dense(output_dim, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def compile_and_train(
        self,
        model: Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        use_mlflow: bool = True
    ) -> Dict[str, List[float]]:
        """
        Compile and train model with callbacks.

        Args:
            model: Keras model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            use_mlflow: Whether to log to MLflow

        Returns:
            Training history
        """
        self.model = model

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        loss = self._get_loss_function()
        metrics = self._get_metrics()

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Callbacks
        callback_list = self._setup_callbacks()

        # MLflow integration
        if use_mlflow:
            mlflow.tensorflow.autolog()
            mlflow.start_run()

        # Train
        logger.info("Starting training...")
        self.history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )

        if use_mlflow:
            mlflow.end_run()

        logger.info("Training completed")
        return self.history.history

    def _get_loss_function(self):
        """Get loss function based on config."""
        task = self.config.get('task', 'regression')

        if task == 'regression':
            return 'mse'
        elif task == 'binary_classification':
            return 'binary_crossentropy'
        elif task == 'multiclass_classification':
            return 'categorical_crossentropy'
        else:
            return 'mse'

    def _get_metrics(self) -> List[str]:
        """Get metrics based on task."""
        task = self.config.get('task', 'regression')

        if task == 'regression':
            return ['mae', 'mse', tf.keras.metrics.RootMeanSquaredError()]
        else:
            return ['accuracy', tf.keras.metrics.AUC()]

    def _setup_callbacks(self) -> List[callbacks.Callback]:
        """Setup training callbacks."""
        callback_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),

            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),

            # Model checkpoint
            callbacks.ModelCheckpoint(
                filepath='models/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),

            # TensorBoard
            callbacks.TensorBoard(
                log_dir='logs',
                histogram_freq=1,
                write_graph=True
            ),

            # CSV logger
            callbacks.CSVLogger('logs/training.log'),

            # Terminate on NaN
            callbacks.TerminateOnNaN(),
        ]

        return callback_list

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

    def save_model(self, path: str):
        """Save model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model."""
        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")


class EnsembleModel:
    """
    Ensemble of multiple Keras models for improved performance.
    """

    def __init__(self, models: List[Model]):
        self.models = models
        self.num_models = len(models)

    def predict(self, X: np.ndarray, method: str = 'average') -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: Input features
            method: Ensemble method ('average', 'weighted', 'voting')

        Returns:
            Ensemble predictions
        """
        predictions = [model.predict(X) for model in self.models]

        if method == 'average':
            return np.mean(predictions, axis=0)
        elif method == 'weighted':
            # Weight by model performance (simplified)
            weights = np.ones(self.num_models) / self.num_models
            return np.average(predictions, axis=0, weights=weights)
        else:
            return np.mean(predictions, axis=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        predictions = self.predict(X)

        from sklearn.metrics import mean_squared_error, mean_absolute_error

        metrics = {
            'mse': mean_squared_error(y, predictions),
            'mae': mean_absolute_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions))
        }

        logger.info(f"Ensemble metrics: {metrics}")
        return metrics


def create_custom_model(config: Dict[str, Any]) -> Model:
    """
    Factory function to create model based on configuration.

    Args:
        config: Model configuration

    Returns:
        Compiled Keras model
    """
    model_type = config.get('model_type', 'residual')
    input_dim = config['input_dim']
    output_dim = config['output_dim']

    builder = AdvancedKerasModel(config)

    if model_type == 'residual':
        model = builder.build_residual_network(
            input_dim=input_dim,
            output_dim=output_dim,
            num_blocks=config.get('num_blocks', 3),
            hidden_dim=config.get('hidden_dim', 256),
            dropout_rate=config.get('dropout_rate', 0.3)
        )
    elif model_type == 'attention':
        model = builder.build_attention_network(
            input_dim=input_dim,
            output_dim=output_dim,
            embed_dim=config.get('embed_dim', 256),
            num_heads=config.get('num_heads', 4),
            ff_dim=config.get('ff_dim', 512),
            num_transformer_blocks=config.get('num_transformer_blocks', 2)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


if __name__ == "__main__":
    # Example usage
    config = {
        'model_type': 'residual',
        'input_dim': 20,
        'output_dim': 1,
        'num_blocks': 3,
        'hidden_dim': 256,
        'dropout_rate': 0.3,
        'task': 'regression'
    }

    # Create and train model
    model = create_custom_model(config)
    print(model.summary())

    # Dummy data
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randn(1000, 1)
    X_val = np.random.randn(200, 20)
    y_val = np.random.randn(200, 1)

    builder = AdvancedKerasModel(config)
    history = builder.compile_and_train(
        model, X_train, y_train, X_val, y_val,
        epochs=10, batch_size=32, use_mlflow=False
    )

    logger.info("TensorFlow model training completed")
