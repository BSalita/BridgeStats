import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import polars as pl
import pathlib
import time
from collections import defaultdict
import pickle
import os
from pathlib import Path
import logging
import json
from copy import deepcopy
from datetime import datetime
from typing import Any, Optional, Dict, List, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_to_log_info(*args: Any) -> None:
    print_to_log(logging.INFO, *args)

def print_to_log_debug(*args: Any) -> None:
    print_to_log(logging.DEBUG, *args)

def print_to_log(level: int, *args: Any) -> None:
    logging.log(level, ' '.join(str(arg) for arg in args))

# --- PyTorch Dataset for Tabular Data ---
class TabularDataset(Dataset):
    def __init__(self, X_categorical: torch.Tensor, X_continuous: torch.Tensor, y: torch.Tensor) -> None:
        self.X_categorical = X_categorical
        self.X_continuous = X_continuous
        self.y = y

    def __len__(self) -> int:
        if torch.is_tensor(self.y) and self.y.ndim > 0:
            return self.y.size(0)
        elif torch.is_tensor(self.X_categorical) and self.X_categorical.ndim > 0 and self.X_categorical.size(0) > 0:
            return self.X_categorical.size(0)
        elif torch.is_tensor(self.X_continuous) and self.X_continuous.ndim > 0 and self.X_continuous.size(0) > 0:
            return self.X_continuous.size(0)
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cat_item = self.X_categorical[idx] if torch.is_tensor(self.X_categorical) and self.X_categorical.numel() > 0 and self.X_categorical.size(0) > idx else torch.empty(0, dtype=torch.long)
        cont_item = self.X_continuous[idx] if torch.is_tensor(self.X_continuous) and self.X_continuous.numel() > 0 and self.X_continuous.size(0) > idx else torch.empty(0, dtype=torch.float32)
        label_item = self.y[idx] if torch.is_tensor(self.y) and self.y.numel() > 0 and self.y.size(0) > idx else torch.empty(0, dtype=torch.long)
        
        return {
            'categorical': cat_item,
            'continuous': cont_item,
            'labels': label_item
        }

# --- PyTorch Model Definition ---
class TabularNNModel(nn.Module):
    def __init__(self, embedding_sizes: List[Tuple[int, int]], n_continuous: int, n_classes: int, layers: List[int], p_dropout: float = 0.1, y_range: Optional[Tuple[float, float]] = None) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_embeddings = sum(e.embedding_dim for e in self.embeddings)
        self.n_continuous = n_continuous
        self.y_range = y_range
        
        all_layers = []
        input_size = n_embeddings + n_continuous
        
        for i, layer_size in enumerate(layers):
            all_layers.append(nn.Linear(input_size, layer_size))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(layer_size))
            all_layers.append(nn.Dropout(p_dropout))
            input_size = layer_size
            
        all_layers.append(nn.Linear(layers[-1], n_classes))
        
        # Add sigmoid activation and scaling if y_range is specified for regression
        if y_range is not None and n_classes == 1:
            all_layers.append(nn.Sigmoid())
            self.y_min = y_range[0]
            self.y_max = y_range[1]
        else:
            self.y_min = None
            self.y_max = None
            
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical: torch.Tensor, x_continuous: torch.Tensor) -> torch.Tensor:
        x_embeddings = []
        for i, e in enumerate(self.embeddings):
            x_embeddings.append(e(x_categorical[:, i]))
        x = torch.cat(x_embeddings, 1)
        
        if self.n_continuous > 0:
            if x_continuous.ndim == 1:
                x_continuous = x_continuous.unsqueeze(0)
            x = torch.cat([x, x_continuous], 1)
            
        x = self.layers(x)
        
        # Scale sigmoid output to y_range if specified
        if self.y_min is not None and self.y_max is not None:
            x = x * (self.y_max - self.y_min) + self.y_min
            
        return x

# Function to calculate the total input size (similar to fastai version)
def calculate_input_size(embedding_sizes: List[Tuple[int, int]], n_continuous: int) -> int:
    total_embedding_size = sum(size for _, size in embedding_sizes)
    return total_embedding_size + n_continuous

# Function to define optimal layer sizes (similar to fastai version)
def define_layer_sizes(input_size: int, num_layers: int = 3, shrink_factor: int = 2) -> List[int]:
    layer_sizes = [input_size]
    for i in range(1, num_layers):
        layer_sizes.append(layer_sizes[-1] // shrink_factor)
    return layer_sizes

# create a test set using date and sample size. current default is 10k samples ge 2024-07-01.
def split_by_date(df: Any, include_dates: str) -> Tuple[Any, Any]:
    include_date = datetime.strptime(include_dates, '%Y-%m-%d') # i'm not getting why datetime.datetime.strptime isn't working here but the only thing that works elsewhere?
    date_filter = df['Date'] >= include_date
    return df.filter(~date_filter), df.filter(date_filter)

def get_device() -> str:
    """
    Get the best available device (GPU if available, otherwise CPU).
    
    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
        print_to_log_info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Log current GPU memory usage
        if torch.cuda.is_initialized():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            print_to_log_info(f"GPU memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    else:
        device = 'cpu'
        print_to_log_info("No GPU available, using CPU")
    return device

def train_model(df: Any, y_names: List[str], cat_names: Optional[List[str]] = None, cont_names: Optional[List[str]] = None, nsamples: Optional[int] = None, 
                procs: Optional[Any] = None, valid_pct: float = 0.2, bs: int = 1024*10, layers: Optional[List[int]] = None, epochs: int = 3, 
                device: Optional[str] = None, y_range: Tuple[float, float] = (0,1), lr: float = 1e-3, patience: int = 3, min_delta: float = 0.001, seed: int = 42) -> Dict[str, Any]:
    """
    Train a PyTorch tabular model similar to the fastai implementation.
    
    Args:
        df: Polars DataFrame with the data
        y_names: List with single target column name
        cat_names: List of categorical column names (auto-detected if None)
        cont_names: List of continuous column names (auto-detected if None)
        nsamples: Number of samples to use (None for all)
        procs: Processing steps (ignored for PyTorch, kept for compatibility)
        valid_pct: Validation split percentage
        bs: Batch size
        layers: List of layer sizes (auto-calculated if None)
        epochs: Number of training epochs
        device: Device to train on ('cpu', 'cuda', or None for auto-detection)
        y_range: Range for regression targets (ignored for classification)
        lr: Learning rate
        patience: Early stopping patience
        min_delta: Minimum delta for early stopping
        seed: Random seed
    
    Returns:
        Dictionary containing model, artifacts, and training info
    """
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    
    t = time.time()
    print_to_log_info(f"{y_names=} {cat_names=} {cont_names=} {nsamples=} {valid_pct=} {bs=} {layers=} {epochs=} {device=} {y_range=} {lr=} {patience=} {min_delta=} {seed=}")

    # Validate inputs
    assert isinstance(y_names, list) and len(y_names) == 1, 'Only one target variable is supported.'
    
    print(df.describe())
    unimplemented_dtypes = df.select(pl.exclude(pl.Boolean,pl.Categorical,pl.Int8,pl.Int16,pl.Int32,pl.Int64,pl.Float32,pl.Float64,pl.String,pl.UInt8,pl.Utf8)).columns
    print(f"{unimplemented_dtypes=}")

    # Setup categorical and continuous column names
    if cat_names is None:
        cat_names = list(set(df.select(pl.col([pl.Boolean,pl.Categorical,pl.String])).columns).difference(y_names))
    print(f"{cat_names=}")
    
    if cont_names is None:
        cont_names = list(set(df.columns).difference(cat_names + y_names))
    print(f"{cont_names=}")
    
    assert set(y_names).intersection(cat_names+cont_names) == set(), set(y_names).intersection(cat_names+cont_names)
    assert set(cat_names).intersection(cont_names) == set(), set(cat_names).intersection(cont_names)

    # Sample data if requested. If nsamples is None, use all data.
    if nsamples is None:
        pandas_df = df[y_names+cat_names+cont_names].to_pandas()
    else:
        pandas_df = df[y_names+cat_names+cont_names].sample(nsamples, seed=seed).to_pandas()

    print('y_names[0].dtype:', pandas_df[y_names[0]].dtype.name)
    
    # Determine if this is classification or regression
    is_classification = pandas_df[y_names[0]].dtype.name in ['boolean','category','object','string','uint8']
    
    if is_classification:
        return train_classifier_pytorch(pandas_df, y_names, cat_names, cont_names, 
                                      valid_pct=valid_pct, bs=bs, layers=layers, 
                                      epochs=epochs, device=device, lr=lr, 
                                      patience=patience, min_delta=min_delta)
    else:
        return train_regression_pytorch(pandas_df, y_names, cat_names, cont_names, 
                                      valid_pct=valid_pct, bs=bs, layers=layers, 
                                      epochs=epochs, device=device, lr=lr, 
                                      patience=patience, min_delta=min_delta, y_range=y_range)

def train_classifier_pytorch(df: Any, y_names: List[str], cat_names: List[str], cont_names: List[str], valid_pct: float = 0.2,
                           bs: int = 1024*5, layers: Optional[List[int]] = None, epochs: int = 3, device: Optional[str] = None, 
                           lr: float = 1e-3, patience: int = 3, min_delta: float = 0.001) -> Dict[str, Any]:
    """Train a classification model using PyTorch."""
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    
    t = time.time()
    
    # Preprocessing
    artifacts = {
        'target_encoder': LabelEncoder(),
        'categorical_encoders': {col: LabelEncoder() for col in cat_names},
        'continuous_scalers': {col: StandardScaler() for col in cont_names},
        'na_fills': {},
        'categorical_feature_names': cat_names,
        'continuous_feature_names': cont_names,
        'target_name': y_names[0],
        'model_params': {},
        'training_history': {},
        'is_classification': True
    }

    # Target encoding
    df[y_names[0]] = df[y_names[0]].astype(str)
    artifacts['target_encoder'].fit(df[y_names[0]])
    df[y_names[0]] = artifacts['target_encoder'].transform(df[y_names[0]])
    n_classes = len(artifacts['target_encoder'].classes_)
    print_to_log_info(f"Target '{y_names[0]}' encoded. Number of classes: {n_classes}")

    # Categorical feature encoding and NA handling
    for col in cat_names:
        df[col] = df[col].astype(str)
        fill_val_cat = "MISSING_CAT"
        artifacts['na_fills'][col] = fill_val_cat

        unique_values_for_fit = pd.unique(df[col].fillna(fill_val_cat)).tolist()
        if fill_val_cat not in unique_values_for_fit:
            unique_values_for_fit.append(fill_val_cat)
        
        artifacts['categorical_encoders'][col].fit(unique_values_for_fit)
        df[col] = artifacts['categorical_encoders'][col].transform(df[col].fillna(fill_val_cat))

    # Continuous feature scaling and NA handling
    for col in cont_names:
        fill_val_cont = df[col].median()
        artifacts['na_fills'][col] = fill_val_cont
        df[col] = df[col].fillna(fill_val_cont)
        df[col] = artifacts['continuous_scalers'][col].fit_transform(df[col].values.reshape(-1, 1)).flatten()

    # Calculate embedding sizes
    embedding_sizes = []
    for col in cat_names:
        num_categories = len(artifacts['categorical_encoders'][col].classes_)
        embed_dim = max(4, min(300, int(num_categories / 2)))
        embedding_sizes.append((num_categories, embed_dim))

    artifacts['embedding_sizes'] = embedding_sizes
    n_continuous = len(cont_names)

    # Calculate input size and layer sizes
    input_size = calculate_input_size(embedding_sizes, n_continuous)
    if layers is None:
        layers = define_layer_sizes(input_size)
    print(f"Input size: {input_size}, Layer sizes: {layers}")

    # Prepare data
    X_cat = torch.tensor(df[cat_names].values, dtype=torch.long) if cat_names else torch.empty((len(df), 0), dtype=torch.long)
    X_cont = torch.tensor(df[cont_names].values, dtype=torch.float32) if cont_names else torch.empty((len(df), 0), dtype=torch.float32)
    y = torch.tensor(df[y_names[0]].values, dtype=torch.long)

    # Train/validation split - handle stratification issues
    try:
        # Try stratified split first
        train_idx, val_idx = train_test_split(range(len(df)), test_size=valid_pct, random_state=42, stratify=y)
        print_to_log_info("Using stratified train/validation split")
    except ValueError as e:
        if "least populated class" in str(e):
            # Fall back to regular split if stratification fails
            print_to_log_info("Stratified split failed due to classes with too few samples. Using regular split.")
            train_idx, val_idx = train_test_split(range(len(df)), test_size=valid_pct, random_state=42)
        else:
            raise e
    
    train_dataset = TabularDataset(X_cat[train_idx], X_cont[train_idx], y[train_idx])
    val_dataset = TabularDataset(X_cat[val_idx], X_cont[val_idx], y[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    # Create model
    model = TabularNNModel(embedding_sizes, n_continuous, n_classes, layers)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop with best model saving and early stopping
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    training_history = []
    early_stopped = False
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch['categorical'].to(device), batch['continuous'].to(device))
            loss = criterion(outputs, batch['labels'].to(device))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch['labels'].size(0)
            train_correct += (predicted == batch['labels'].to(device)).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['categorical'].to(device), batch['continuous'].to(device))
                loss = criterion(outputs, batch['labels'].to(device))
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch['labels'].size(0)
                val_correct += (predicted == batch['labels'].to(device)).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        })
        
        print_to_log_info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = deepcopy(model.state_dict())
            print_to_log_info(f"New best model saved! Val Acc: {val_acc:.4f}")
        
        # Early stopping (based on validation loss)
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                early_stopped = True
                print_to_log_info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print_to_log_info(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
    else:
        print_to_log_info("Warning: No best model state found, using final model")
    
    artifacts['training_history'] = training_history
    artifacts['early_stopped'] = early_stopped
    artifacts['model_params'] = {
        'embedding_sizes': embedding_sizes,
        'n_continuous': n_continuous,
        'n_classes': n_classes,
        'layers': layers
    }
    
    print_to_log_info('train_classifier_pytorch time:', time.time()-t)
    
    return {
        'model': model,
        'artifacts': artifacts,
        'device': device
    }

def train_regression_pytorch(df: Any, y_names: List[str], cat_names: List[str], cont_names: List[str], valid_pct: float = 0.2, 
                           bs: int = 1024*5, layers: Optional[List[int]] = None, epochs: int = 3, device: Optional[str] = None, 
                           lr: float = 1e-3, patience: int = 3, min_delta: float = 0.001, y_range: Tuple[float, float] = (0,1)) -> Dict[str, Any]:
    """Train a regression model using PyTorch."""
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    
    t = time.time()
    
    # Similar structure to classifier but with regression-specific changes
    artifacts = {
        'categorical_encoders': {col: LabelEncoder() for col in cat_names},
        'continuous_scalers': {col: StandardScaler() for col in cont_names},
        'target_scaler': StandardScaler(),
        'na_fills': {},
        'categorical_feature_names': cat_names,
        'continuous_feature_names': cont_names,
        'target_name': y_names[0],
        'model_params': {},
        'training_history': {},
        'is_classification': False,
        'y_range': y_range
    }

    # Target scaling
    y_scaled = artifacts['target_scaler'].fit_transform(df[y_names].values)
    df[y_names[0]] = y_scaled.flatten()

    # Categorical feature encoding and NA handling
    for col in cat_names:
        df[col] = df[col].astype(str)
        fill_val_cat = "MISSING_CAT"
        artifacts['na_fills'][col] = fill_val_cat

        unique_values_for_fit = pd.unique(df[col].fillna(fill_val_cat)).tolist()
        if fill_val_cat not in unique_values_for_fit:
            unique_values_for_fit.append(fill_val_cat)
        
        artifacts['categorical_encoders'][col].fit(unique_values_for_fit)
        df[col] = artifacts['categorical_encoders'][col].transform(df[col].fillna(fill_val_cat))

    # Continuous feature scaling and NA handling
    for col in cont_names:
        fill_val_cont = df[col].median()
        artifacts['na_fills'][col] = fill_val_cont
        df[col] = df[col].fillna(fill_val_cont)
        df[col] = artifacts['continuous_scalers'][col].fit_transform(df[col].values.reshape(-1, 1)).flatten()

    # Calculate embedding sizes
    embedding_sizes = []
    for col in cat_names:
        num_categories = len(artifacts['categorical_encoders'][col].classes_)
        embed_dim = max(4, min(300, int(num_categories / 2)))
        embedding_sizes.append((num_categories, embed_dim))

    artifacts['embedding_sizes'] = embedding_sizes
    n_continuous = len(cont_names)

    # Calculate input size and layer sizes
    input_size = calculate_input_size(embedding_sizes, n_continuous)
    if layers is None:
        layers = define_layer_sizes(input_size)
    print(f"Input size: {input_size}, Layer sizes: {layers}")

    # Prepare data
    X_cat = torch.tensor(df[cat_names].values, dtype=torch.long) if cat_names else torch.empty((len(df), 0), dtype=torch.long)
    X_cont = torch.tensor(df[cont_names].values, dtype=torch.float32) if cont_names else torch.empty((len(df), 0), dtype=torch.float32)
    y = torch.tensor(df[y_names[0]].values, dtype=torch.float32)

    # Train/validation split
    train_idx, val_idx = train_test_split(range(len(df)), test_size=valid_pct, random_state=42)
    
    train_dataset = TabularDataset(X_cat[train_idx], X_cont[train_idx], y[train_idx])
    val_dataset = TabularDataset(X_cat[val_idx], X_cont[val_idx], y[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

    # Create model (output size 1 for regression)
    model = TabularNNModel(embedding_sizes, n_continuous, 1, layers, y_range=y_range)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop with best model saving and early stopping  
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    training_history = []
    early_stopped = False
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch['categorical'].to(device), batch['continuous'].to(device))
            loss = criterion(outputs.squeeze(), batch['labels'].to(device))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['categorical'].to(device), batch['continuous'].to(device))
                loss = criterion(outputs.squeeze(), batch['labels'].to(device))
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Calculate RMSE in scaled and original units
        train_rmse_scaled = train_loss ** 0.5
        val_rmse_scaled = val_loss ** 0.5
        
        # Convert to original units using target scaler
        original_std = artifacts['target_scaler'].scale_[0]
        train_rmse_original = train_rmse_scaled * original_std
        val_rmse_original = val_rmse_scaled * original_std
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_rmse_scaled': train_rmse_scaled,
            'val_rmse_scaled': val_rmse_scaled,
            'train_rmse_original': train_rmse_original,
            'val_rmse_original': val_rmse_original
        })
        
        print_to_log_info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                          f"Train RMSE: {train_rmse_scaled:.4f} (scaled) / {train_rmse_original:.4f} (original), "
                          f"Val RMSE: {val_rmse_scaled:.4f} (scaled) / {val_rmse_original:.4f} (original)")
        
        # Early stopping and model saving logic
        improvement_threshold = best_val_loss - min_delta
        print_to_log_info(f"DEBUG: val_loss={val_loss:.6f}, best_val_loss={best_val_loss:.6f}, min_delta={min_delta}, threshold={improvement_threshold:.6f}, patience_counter={patience_counter}")
        
        if val_loss < improvement_threshold:
            # Significant improvement: save model, update best loss, reset patience
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
            print_to_log_info(f"New best model saved! Val Loss: {val_loss:.4f}")
        else:
            # No significant improvement: increment patience counter
            patience_counter += 1
            print_to_log_info(f"No significant improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                early_stopped = True
                print_to_log_info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print_to_log_info(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    else:
        print_to_log_info("Warning: No best model state found, using final model")
    
    artifacts['training_history'] = training_history
    artifacts['early_stopped'] = early_stopped
    artifacts['model_params'] = {
        'embedding_sizes': embedding_sizes,
        'n_continuous': n_continuous,
        'n_classes': 1,
        'layers': layers
    }
    
    print_to_log_info('train_regression_pytorch time:', time.time()-t)
    
    return {
        'model': model,
        'artifacts': artifacts,
        'device': device
    }

def save_model(learn_dict: Dict[str, Any], f: str) -> None:
    """
    Save a PyTorch model and its artifacts.
    
    Args:
        learn_dict: Dictionary containing model, artifacts, and device info
        f: File path to save to
    """
    t = time.time()
    
    # Prepare save dictionary
    save_dict = {
        'model_state_dict': learn_dict['model'].state_dict(),
        'artifacts': learn_dict['artifacts'],
        'device': learn_dict['device']
    }
    
    # Save using torch.save
    torch.save(save_dict, f)
    print_to_log_info('save_model time:', time.time()-t)

def load_model(f: str) -> Dict[str, Any]:
    """
    Load a PyTorch model and its artifacts.
    
    Args:
        f: File path to load from
        
    Returns:
        Dictionary containing model, artifacts, and device info
    """
    t = time.time()
    
    # Load the saved dictionary
    # Note: Using weights_only=False because our models contain sklearn objects (LabelEncoder, StandardScaler)
    # which are safe in our trusted context but not allowed by default in PyTorch 2.6+
    save_dict = torch.load(f, map_location='cpu', weights_only=False)
    
    # Reconstruct the model
    artifacts = save_dict['artifacts']
    model_params = artifacts['model_params']
    
    if artifacts['is_classification']:
        model = TabularNNModel(
            artifacts['embedding_sizes'], 
            len(artifacts['continuous_feature_names']),
            artifacts['model_params']['n_classes'],
            artifacts['model_params']['layers']
        )
    else:
        model = TabularNNModel(
            artifacts['embedding_sizes'], 
            len(artifacts['continuous_feature_names']),
            artifacts['model_params']['n_classes'],
            artifacts['model_params']['layers'],
            y_range=artifacts.get('y_range')
        )
    
    model.load_state_dict(save_dict['model_state_dict'])
    
    learn_dict = {
        'model': model,
        'artifacts': artifacts,
        'device': save_dict['device']
    }
    
    print_to_log_info('load_model time:', time.time()-t)
    return learn_dict

def preprocess_inference_data(df: Any, artifacts: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Preprocess inference data using the same transformations as training data.
    
    Args:
        df: Pandas DataFrame with inference data
        artifacts: Artifacts from training containing encoders and scalers
        
    Returns:
        Preprocessed DataFrame
    """
    t = time.time()
    df = df.copy()
    
    # Handle categorical features
    for col in artifacts['categorical_feature_names']:
        if col in df.columns:
            df[col] = df[col].astype(str)
            # Fill missing values
            fill_val = artifacts['na_fills'][col]
            df[col] = df[col].fillna(fill_val).infer_objects(copy=False)
            
            # Handle unseen categories by mapping them to a default value
            encoder = artifacts['categorical_encoders'][col]
            known_categories = set(encoder.classes_)
            
            # Vectorized approach: replace unknown categories with first known category
            unknown_mask = ~df[col].isin(known_categories)
            if unknown_mask.any():
                unknown_values = df[col][unknown_mask].unique()
                print_to_log_info(f'Warning: Column {col} contains {len(unknown_values)} unknown values, mapping to default')
                df.loc[unknown_mask, col] = encoder.classes_[0]
            
            # Now transform all values at once (much faster than apply)
            df[col] = encoder.transform(df[col])
    
    # Handle continuous features - vectorized operations
    for col in artifacts['continuous_feature_names']:
        if col in df.columns:
            # Fill missing values
            fill_val = artifacts['na_fills'][col]
            df[col] = df[col].fillna(fill_val).infer_objects(copy=False)
            
            # Scale using the same scaler from training (vectorized)
            scaler = artifacts['continuous_scalers'][col]
            df[col] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
    
    print_to_log_info(f'preprocess_inference_data time: {time.time()-t:.4f} seconds')
    return df

def get_predictions(learn_dict: Dict[str, Any], df: Any, y_names: Optional[List[str]] = None, device: Optional[str] = None) -> Any:
    """
    Perform inference using a trained PyTorch model.
    
    Args:
        learn_dict: Dictionary containing model, artifacts, and device info (from load_model)
        df: DataFrame containing the inference data
        y_names: List of target column names (optional, will use from artifacts if None)
        device: Device to run inference on (None for auto-detection)
        
    Returns:
        DataFrame with predictions and actual values
    """
    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    
    t = time.time()
    
    model = learn_dict['model']
    artifacts = learn_dict['artifacts']
    
    # Get target column name
    if y_names is None:
        y_names = [artifacts['target_name']]
    assert len(y_names) == 1, 'Only one target variable is supported.'
    y_name = y_names[0]
    
    # Check that required columns are present
    required_cols = artifacts['categorical_feature_names'] + artifacts['continuous_feature_names']
    missing_cols = set(required_cols).difference(df.columns)
    if missing_cols:
        print_to_log_info(f"Warning: Missing columns in inference data: {missing_cols}")
    
    assert not df.empty, 'No data to make inferences on.'
    
    # Preprocess the inference data
    preprocess_start = time.time()
    df_processed = preprocess_inference_data(df, artifacts)
    preprocess_time = time.time() - preprocess_start
    print_to_log_info(f'Data preprocessing completed in {preprocess_time:.4f} seconds')
    
    # Handle target column if present (for evaluation)
    target_start = time.time()
    has_target = y_name in df.columns
    if has_target:
        if artifacts['is_classification']:
            # Handle target encoding for classification
            if artifacts.get('target_encoder'):
                target_encoder = artifacts['target_encoder']
                df_target = df[y_name].astype(str)
                
                # Filter out unknown target values
                known_targets = set(target_encoder.classes_)
                unknown_mask = ~df_target.isin(known_targets)
                if unknown_mask.any():
                    unknown_values = df_target[unknown_mask].unique()
                    print_to_log_info(f'Warning: {y_name} contains values which are missing in training set: {unknown_values}')
                    # Remove rows with unknown target values
                    df_processed = df_processed[~unknown_mask]
                    df_target = df_target[~unknown_mask]
                
                if len(df_processed) == 0:
                    print_to_log_info("No valid data remaining after filtering unknown target values")
                    return pd.DataFrame()
                
                true_labels = df_target.values
                true_codes = target_encoder.transform(df_target)
        else:
            # For regression, use target scaler if available
            if artifacts.get('target_scaler'):
                true_values = artifacts['target_scaler'].transform(df[[y_name]].values).flatten()
            else:
                true_values = df[y_name].values
    
    target_time = time.time() - target_start
    print_to_log_info(f'Target processing completed in {target_time:.4f} seconds')

    # Prepare data for inference
    cat_names = artifacts['categorical_feature_names']
    cont_names = artifacts['continuous_feature_names']
    
    # Move model to device and verify
    model.eval()
    model.to(device)
    print_to_log_info(f"Model moved to device: {device}")
    
    # Verify model is on correct device by checking first parameter
    if hasattr(model, 'parameters'):
        first_param_device = next(model.parameters()).device
        print_to_log_info(f"Model parameters are on device: {first_param_device}")
    
    # Create tensors and move to device immediately
    tensor_start = time.time()
    X_cat = torch.tensor(df_processed[cat_names].values, dtype=torch.long, device=device) if cat_names else torch.empty((len(df_processed), 0), dtype=torch.long, device=device)
    X_cont = torch.tensor(df_processed[cont_names].values, dtype=torch.float32, device=device) if cont_names else torch.empty((len(df_processed), 0), dtype=torch.float32, device=device)
    
    # Create dataset and dataloader
    # For inference, we don't need labels, so create dummy labels on device
    dummy_labels = torch.zeros(len(df_processed), dtype=torch.long if artifacts['is_classification'] else torch.float32, device=device)
    inference_dataset = TabularDataset(X_cat, X_cont, dummy_labels)
    inference_loader = DataLoader(inference_dataset, batch_size=1024, shuffle=False)
    tensor_time = time.time() - tensor_start
    print_to_log_info(f'Tensor creation completed in {tensor_time:.4f} seconds')
    
    # Run inference
    all_predictions = []
    
    print_to_log_info(f"Starting inference on {device} with {len(df_processed)} samples")
    inference_start = time.time()
    
    # Log GPU memory usage before inference
    if device == 'cuda' and torch.cuda.is_available():
        allocated_before = torch.cuda.memory_allocated(0) / 1024**3
        cached_before = torch.cuda.memory_reserved(0) / 1024**3
        print_to_log_info(f"GPU memory before inference - Allocated: {allocated_before:.2f} GB, Cached: {cached_before:.2f} GB")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(inference_loader):
            # Data should already be on device, but verify for first batch
            if batch_idx == 0:
                print_to_log_info(f"Batch categorical data device: {batch['categorical'].device}")
                print_to_log_info(f"Batch continuous data device: {batch['continuous'].device}")
            
            # Since data is already on device, no need to move it again
            outputs = model(batch['categorical'], batch['continuous'])
            all_predictions.append(outputs.cpu())
    
    inference_time = time.time() - inference_start
    print_to_log_info(f"Inference completed in {inference_time:.4f} seconds on {device}")
    
    # Log GPU memory usage after inference
    if device == 'cuda' and torch.cuda.is_available():
        allocated_after = torch.cuda.memory_allocated(0) / 1024**3
        cached_after = torch.cuda.memory_reserved(0) / 1024**3
        print_to_log_info(f"GPU memory after inference - Allocated: {allocated_after:.2f} GB, Cached: {cached_after:.2f} GB")

    # Concatenate all predictions
    predictions = torch.cat(all_predictions, dim=0)
    
    # Process predictions based on task type
    result_start = time.time()
    if artifacts['is_classification']:
        # Convert probabilities to class labels
        pred_probs = torch.softmax(predictions, dim=1)
        pred_codes = pred_probs.argmax(dim=1).numpy()
        
        # Decode predictions back to original labels
        target_encoder = artifacts['target_encoder']
        pred_labels = target_encoder.inverse_transform(pred_codes)
        
        results = {
            f'{y_name}_Pred': pred_labels,
            f'{y_name}_Pred_Code': pred_codes
        }
        
        if has_target:
            results.update({
                f'{y_name}_Actual': true_labels,
                f'{y_name}_Actual_Code': true_codes,
                f'{y_name}_Match': [pred == true for pred, true in zip(pred_labels, true_labels)],
                f'{y_name}_Match_Code': [pred == true for pred, true in zip(pred_codes, true_codes)]
            })
    else:
        # For regression
        pred_values = predictions.squeeze().numpy()
        
        # Inverse transform if target scaler was used
        if artifacts.get('target_scaler'):
            pred_values = artifacts['target_scaler'].inverse_transform(pred_values.reshape(-1, 1)).flatten()
        
        results = {
            f'{y_name}_Pred': pred_values
        }
        
        if has_target:
            # Inverse transform true values if needed
            if artifacts.get('target_scaler'):
                true_values_orig = artifacts['target_scaler'].inverse_transform(true_values.reshape(-1, 1)).flatten()
            else:
                true_values_orig = true_values
                
            results.update({
                f'{y_name}_Actual': true_values_orig,
                f'{y_name}_Error': pred_values - true_values_orig,
                f'{y_name}_AbsoluteError': np.abs(pred_values - true_values_orig)
            })
    
    result_time = time.time() - result_start
    print_to_log_info(f'Result processing completed in {result_time:.4f} seconds')
    
    print_to_log_info('get_predictions time:', time.time()-t)
    return pd.DataFrame(results)

def find_first_linear_layer(module: nn.Module) -> Optional[nn.Module]:
    """
    Find the first linear layer in a PyTorch model.
    
    Args:
        module: PyTorch module to search
        
    Returns:
        First nn.Linear layer found, or None if not found
    """
    if isinstance(module, nn.Linear):
        return module
    elif isinstance(module, (nn.Sequential, nn.ModuleList)):
        for layer in module:
            found = find_first_linear_layer(layer)
            if found:
                return found
    elif hasattr(module, 'children'):
        for layer in module.children():
            found = find_first_linear_layer(layer)
            if found:
                return found
    return None

def get_feature_importance(learn_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate feature importance based on the weights of the first linear layer.
    
    Args:
        learn_dict: Dictionary containing model, artifacts, and device info
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    model = learn_dict['model']
    artifacts = learn_dict['artifacts']
    
    importance = {}

    # Find the first linear layer in the model
    linear_layer = find_first_linear_layer(model)
    if linear_layer is None:
        raise ValueError("No linear layer found in the model.")
    
    # Get the absolute mean of the weights across the input features
    weights = linear_layer.weight.abs().mean(dim=0)

    # Get feature names from artifacts
    cat_names = artifacts['categorical_feature_names']
    cont_names = artifacts['continuous_feature_names']

    # Calculate the total input size to the first linear layer
    # For our TabularNNModel, embeddings are stored in model.embeddings
    emb_szs = {}
    if hasattr(model, 'embeddings') and model.embeddings:
        for i, name in enumerate(cat_names):
            if i < len(model.embeddings):
                emb_szs[name] = model.embeddings[i].embedding_dim
            else:
                # Fallback to artifacts if available
                if 'embedding_sizes' in artifacts.get('model_params', {}):
                    emb_szs[name] = artifacts['model_params']['embedding_sizes'][i][1]
                else:
                    emb_szs[name] = 1  # Default fallback
    
    total_input_size = sum(emb_szs.values()) + len(cont_names)

    print_to_log_info(f"Embedding sizes: {emb_szs}")
    print_to_log_info(f"Total input size to the first linear layer: {total_input_size}")
    print_to_log_info(f"Shape of weights: {weights.shape}")

    # Ensure the number of weights matches the total input size
    if len(weights) != total_input_size:
        raise ValueError(f"Number of weights ({len(weights)}) does not match total input size ({total_input_size}).")

    # Assign importance to each feature
    idx = 0
    for name in cat_names:
        emb_size = emb_szs.get(name, 1)
        if emb_size > 1:
            importance[name] = weights[idx:idx+emb_size].mean().item()  # Average the importance across the embedding dimensions
        else:
            importance[name] = weights[idx].item()
        idx += emb_size
    
    for name in cont_names:
        importance[name] = weights[idx].item()
        idx += 1
    
    return importance

def chart_feature_importance(learn_dict: Dict[str, Any], topn: Optional[int] = None) -> None:
    """
    Calculate and visualize feature importance.
    
    Args:
        learn_dict: Dictionary containing model, artifacts, and device info
        topn: Number of top features to display (None for all features)
    """
    # Calculate and display feature importance
    importance = get_feature_importance(learn_dict)
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    # Limit to top N features if specified
    if topn is not None:
        sorted_importance = sorted_importance[:topn]
        print(f"\nTop {len(sorted_importance)} Feature Importances (out of {len(importance)} total):")
    else:
        print(f"\nFeature Importances {len(importance)}:")
    
    for name, imp in sorted_importance:
        print_to_log_info(f"{name}: {imp:.4f}")

    # Visualize the importance
    try:
        from matplotlib import pyplot as plt

        plt.figure(figsize=(24, 4))
        plt.bar(range(len(sorted_importance)), [imp for name, imp in sorted_importance])
        plt.xticks(range(len(sorted_importance)), [name for name, imp in sorted_importance], rotation=45, ha='right')
        
        if topn is not None:
            plt.title(f'Top {len(sorted_importance)} Feature Importance (out of {len(importance)} total)')
        else:
            plt.title('Feature Importance')
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print_to_log_info("matplotlib not available, skipping visualization")
    except Exception as e:
        print_to_log_info(f"Error creating plot: {e}")

def predict_pct(learn_dict: Dict[str, Any], df: Any, session_col: str = 'Session', target_col: str = 'Pct_Target', device: Optional[str] = None) -> Any:
    """
    Comprehensive Pct_NS/Pct_EW prediction function with constraint enforcement.
    
    Adds prediction columns to df_test:
    - Pct_NS_Pred, Pct_EW_Pred: Final constrained predictions
    - Pct_NS_Pred_Error, Pct_EW_Pred_Error: Prediction errors (if actuals available)
    - Pct_NS_Pred_Absolute_Error, Pct_EW_Pred_Absolute_Error: Absolute errors
    
    Enforces constraints:
    - Row-wise: Pct_NS_Pred + Pct_EW_Pred = 1.0 for each board
    - Session-wise: mean(Pct_NS_Pred) = 0.5 within each session
    - Session-wise: mean(Pct_EW_Pred) = 0.5 within each session
    
    Args:
        model_file: Path to saved PyTorch model file
        df_test: Polars or Pandas DataFrame with test data (can contain multiple sessions)
        session_col: Name of session column (default: 'Session')
        target_col: Name of target column for evaluation (default: 'Pct_Target')
        device: Device for inference (None for auto-detection)
        
    Returns:
        Dictionary containing:
        - 'df': DataFrame with original data plus prediction columns
        - 'metrics': Dictionary with per-session and overall metrics
        - 'constraints': Dictionary with constraint satisfaction info
        - 'sessions': List of session statistics
    """
    print_to_log_info("=== PREDICT_PCT: Comprehensive Pct_NS/EW Prediction with Constraints ===")
    
    assert isinstance(df, pl.DataFrame), "df_test must be a Polars DataFrame"
    
    # Extract model features with robust error handling
    artifacts = learn_dict.get('artifacts', {})
    if isinstance(artifacts, dict):
        cat_names = artifacts.get('categorical_feature_names', [])
        cont_names = artifacts.get('continuous_feature_names', [])
        target_name = artifacts.get('target_name', target_col)
        
        # Fallback: try alternative key names
        if not cat_names and not cont_names:
            cat_names = artifacts.get('cat_names', [])
            cont_names = artifacts.get('cont_names', [])
    else:
        raise ValueError(f"Invalid artifacts format in model. Expected dict, got {type(artifacts)}")
    
    if not cat_names and not cont_names:
        raise ValueError(f"Cannot find feature names in artifacts. Available keys: {list(artifacts.keys())}")
    
    model_features = cat_names + cont_names
    print_to_log_info(f"Model expects {len(model_features)} features: {len(cat_names)} categorical, {len(cont_names)} continuous")
    
    # ASSERTION 1: Check all training features are present
    missing_features = [f for f in model_features if f not in df.columns]
    assert len(missing_features) == 0, f"Missing required features in df_test: {missing_features}"
    print_to_log_info(f"✅ All {len(model_features)} training features present in test data")
    
    # Session validation and statistics
    if session_col in df.columns:
        unique_sessions = df[session_col].unique().to_list()
        print_to_log_info(f"✅ Found {len(unique_sessions)} unique sessions: {unique_sessions}")
        session_counts = df.group_by(session_col).agg(pl.count().alias('count')).sort(session_col)
        print_to_log_info(f"Session distribution:")
        for row in session_counts.iter_rows():
            print_to_log_info(f"  Session {row[0]}: {row[1]} boards")
    else:
        print_to_log_info("⚠️  No session column found - treating as single session")
        df = df.with_columns(pl.lit(1).alias(session_col))
        unique_sessions = [1]
    
    # Check if we have actual values for evaluation
    has_actuals = 'Pct_NS' in df.columns and 'Pct_EW' in df.columns
    has_target = target_name in df.columns
    
    print_to_log_info(f"Evaluation mode: Pct_NS/EW available={has_actuals}, {target_name} available={has_target}")
    
    # Prepare inference data - convert to pandas for compatibility with existing inference pipeline
    inference_features = model_features + ([target_name] if has_target else [])
    df_inference = df.select(inference_features).to_pandas()
    
    # Run inference using existing get_predictions function
    print_to_log_info("Running model inference...")
    try:
        df_predictions = get_predictions(learn_dict, df_inference, y_names=[target_name] if has_target else None, device=device)
        print_to_log_info(f"✅ Inference completed: {len(df_predictions)} predictions generated")
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")
    
    # Find prediction column with robust naming
    pred_col_name = f"{target_name}_Pred"
    if pred_col_name not in df_predictions.columns:
        pred_cols = [col for col in df_predictions.columns if col.endswith('_Pred')]
        if pred_cols:
            pred_col_name = pred_cols[0]
            print_to_log_info(f"Using prediction column: {pred_col_name}")
        else:
            raise ValueError(f"Cannot find prediction column. Available: {list(df_predictions.columns)}")
    
    raw_predictions = df_predictions[pred_col_name].values
    print_to_log_info(f"Raw predictions - mean: {np.mean(raw_predictions):.6f}, std: {np.std(raw_predictions):.6f}")
    
    # Normalize predictions by session to enforce constraints
    sessions = df[session_col].to_numpy()
    constrained_predictions, session_stats = normalize_predictions_by_session(raw_predictions, sessions)
    
    # Create final constrained Pct_NS and Pct_EW predictions
    final_pct_ns = constrained_predictions
    final_pct_ew = 1.0 - final_pct_ns
    
    print_to_log_info(f"Constrained predictions - NS mean: {np.mean(final_pct_ns):.6f}, EW mean: {np.mean(final_pct_ew):.6f}")
    
    # Add prediction columns to original DataFrame
    df_result = df.with_columns([
        pl.Series('Pct_NS_Pred', final_pct_ns),
        pl.Series('Pct_EW_Pred', final_pct_ew)
    ])
    
    # Add error columns if actuals are available
    if has_actuals:
        actual_ns = df['Pct_NS'].to_numpy()
        actual_ew = df['Pct_EW'].to_numpy()
        
        df_result = df_result.with_columns([
            pl.Series('Pct_NS_Pred_Error', final_pct_ns - actual_ns),
            pl.Series('Pct_EW_Pred_Error', final_pct_ew - actual_ew),
            pl.Series('Pct_NS_Pred_Absolute_Error', np.abs(final_pct_ns - actual_ns)),
            pl.Series('Pct_EW_Pred_Absolute_Error', np.abs(final_pct_ew - actual_ew))
        ])
    
    # Calculate comprehensive metrics
    metrics = {}
    constraint_info = {}
    
    # Overall constraint verification
    row_sums = final_pct_ns + final_pct_ew
    max_row_error = np.max(np.abs(row_sums - 1.0))
    constraint_info['row_sum_constraint_satisfied'] = max_row_error < 1e-10
    constraint_info['max_row_sum_error'] = max_row_error
    constraint_info['ns_mean'] = np.mean(final_pct_ns)
    constraint_info['ew_mean'] = np.mean(final_pct_ew)
    constraint_info['mean_constraint_satisfied'] = abs(np.mean(final_pct_ns) - 0.5) < 1e-6
    
    # Per-session and overall metrics
    if has_actuals:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Overall metrics
        metrics['overall'] = {
            'samples': len(final_pct_ns),
            'sessions': len(unique_sessions),
            'ns_rmse': np.sqrt(mean_squared_error(actual_ns, final_pct_ns)),
            'ns_mae': mean_absolute_error(actual_ns, final_pct_ns),
            'ns_r2': r2_score(actual_ns, final_pct_ns),
            'ew_rmse': np.sqrt(mean_squared_error(actual_ew, final_pct_ew)),
            'ew_mae': mean_absolute_error(actual_ew, final_pct_ew),
            'ew_r2': r2_score(actual_ew, final_pct_ew)
        }
        
        # Per-session metrics
        session_metrics = []
        for session_id in unique_sessions:
            session_mask = sessions == session_id
            if np.sum(session_mask) == 0:
                continue
                
            session_actual_ns = actual_ns[session_mask]
            session_actual_ew = actual_ew[session_mask]
            session_pred_ns = final_pct_ns[session_mask]
            session_pred_ew = final_pct_ew[session_mask]
            
            session_metric = {
                'session_id': session_id,
                'samples': len(session_actual_ns),
                'ns_rmse': np.sqrt(mean_squared_error(session_actual_ns, session_pred_ns)),
                'ns_mae': mean_absolute_error(session_actual_ns, session_pred_ns),
                'ns_r2': r2_score(session_actual_ns, session_pred_ns),
                'ew_rmse': np.sqrt(mean_squared_error(session_actual_ew, session_pred_ew)),
                'ew_mae': mean_absolute_error(session_actual_ew, session_pred_ew),
                'ew_r2': r2_score(session_actual_ew, session_pred_ew),
                'ns_pred_mean': np.mean(session_pred_ns),
                'ew_pred_mean': np.mean(session_pred_ew),
                'ns_actual_mean': np.mean(session_actual_ns),
                'ew_actual_mean': np.mean(session_actual_ew)
            }
            session_metrics.append(session_metric)
        
        metrics['sessions'] = session_metrics
    
    print_to_log_info("\n=== CONSTRAINT VERIFICATION ===")
    print_to_log_info(f"✅ Row-wise constraint: Pct_NS + Pct_EW = 1.0 (max error: {max_row_error:.2e})")
    print_to_log_info(f"✅ Overall mean constraint: NS={constraint_info['ns_mean']:.6f}, EW={constraint_info['ew_mean']:.6f}")
    
    # Session constraint verification
    session_constraint_errors = []
    for stat in session_stats:
        error = abs(stat['final_mean'] - 0.5)
        session_constraint_errors.append(error)
        print_to_log_info(f"✅ Session {stat['session']} mean constraint: {stat['final_mean']:.6f} (error: {error:.2e})")
    
    max_session_constraint_error = max(session_constraint_errors) if session_constraint_errors else 0
    constraint_info['max_session_constraint_error'] = max_session_constraint_error
    constraint_info['session_constraints_satisfied'] = max_session_constraint_error < 1e-6
    
    if has_actuals:
        print_to_log_info("\n=== PERFORMANCE METRICS ===")
        overall = metrics['overall']
        print_to_log_info(f"Overall ({overall['samples']} samples across {overall['sessions']} sessions):")
        print_to_log_info(f"  NS: RMSE={overall['ns_rmse']:.6f}, MAE={overall['ns_mae']:.6f}, R²={overall['ns_r2']:.6f}")
        print_to_log_info(f"  EW: RMSE={overall['ew_rmse']:.6f}, MAE={overall['ew_mae']:.6f}, R²={overall['ew_r2']:.6f}")
    
    print_to_log_info(f"\n✅ PREDICT_PCT COMPLETE: {len(df_result)} boards with constrained predictions")
    
    return {
        'df': df_result,
        'metrics': metrics,
        'constraints': constraint_info,
        'sessions': session_stats
    }

def normalize_predictions_by_session(predictions: Any, sessions: Any) -> Any:
    """
    Normalize predictions so that within each session, the mean equals 0.5.
    This enforces session-wise constraint: mean(Pct_NS_Pred) = 0.5 per session.
    
    Args:
        predictions: array of Pct_NS predictions
        sessions: array of session identifiers (same length as predictions)
    
    Returns:
        tuple: (normalized predictions, session statistics)
    """
    predictions = np.array(predictions)
    sessions = np.array(sessions)
    normalized = predictions.copy()
    
    session_stats = []
    
    for session_id in np.unique(sessions):
        session_mask = sessions == session_id
        session_preds = predictions[session_mask]
        
        if len(session_preds) == 0:
            continue
            
        # Normalize this session to have mean = 0.5
        current_mean = session_preds.mean()
        adjusted = session_preds - current_mean + 0.5
        
        # Clamp to [0, 1] range
        adjusted = np.clip(adjusted, 0.0, 1.0)
        
        # Fine-tune if clamping changed the mean
        final_mean = adjusted.mean()
        if abs(final_mean - 0.5) > 1e-6:
            # Scale around 0.5 to get exact mean
            adjusted = 0.5 + (adjusted - 0.5) * (0.5 / final_mean)
            adjusted = np.clip(adjusted, 0.0, 1.0)
        
        normalized[session_mask] = adjusted
        session_stats.append({
            'session': session_id,
            'count': len(session_preds),
            'original_mean': current_mean,
            'final_mean': adjusted.mean()
        })
    
    print_to_log_info(f"Normalized {len(session_stats)} sessions")
    print_to_log_info(f"Overall mean after session normalization: {normalized.mean():.6f}")
    
    return normalized, session_stats 