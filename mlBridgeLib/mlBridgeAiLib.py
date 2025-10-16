
# takes 1h10m/35m to produce df of 48m rows by 2546 columns. File size 33GB. Uses 1.4TB of memory/pagefile.
# Create training data by merging of hand records augmented and board results augmented.

# claude-4-sonnet says: GLOBAL ISSUES IDENTIFIED:
# Issue 12: Memory Usage - 1.4TB memory usage indicates inefficient data loading/processing
# Issue 13: Suboptimal Hyperparameters - batch size 32K may be too large, only 3 epochs may be insufficient
# Issue 14: Poor Error Handling - Most functions lack comprehensive exception handling
# Issue 15: Hardcoded Paths - Many file paths hardcoded rather than parameterized
# Issue 17: Missing Docstrings - Many functions lack proper documentation

# gemini-2.5-flash says: GLOBAL ISSUES IDENTIFIED:
# Issue 16: Inconsistent Categorical Feature Handling - `MLP` model in Cell 21 expects `cat_dims` but `_train_model_core` (Cell 22) initializes it with `None`.
# Issue 18: Lack of GPU Resource Management - No explicit handling or check for GPU memory availability or utilization, which is crucial for large datasets.
# Issue 19: No Experiment Tracking - No system for logging experiments, hyperparameters, and metrics, making reproducibility and comparison difficult.
# Issue 20: No Version Control for Models/Schemas - Models and schemas are saved to a path without explicit versioning, risking overwrites and difficulty reverting.
# Issue 21: Unhandled `y_range` in Inference - The `predict_batches` function clamps predictions with `y_range` but doesn't store this `y_range` in the schema or pass it explicitly to the prediction function.

# gpt-5 says: GLOBAL ISSUES REMAINING:
# - Persist y_range in schema/model stats and load at inference.
# - Inference must build separate X_cont and X_cat; avoid mixing and scaling categoricals as continuous.
# - Unify categorical mapping and scaling via shared helpers to remove duplication.
# - Ensure validate_feature_consistency compares against expected input dim (cont + embeddings), not raw feature list length.
# - Add schema versioning and run manifest (hyperparams, metrics, schema hash, git SHA).
# - Validation should use fixed date cutoff; add early stopping and experiment tracking/versioning.
# - Add OOM-safe training (dynamic batch/grad accumulation) and basic GPU mem checks/logging.
# - Consider calibration plots and binned metrics in analysis.
# - Add robust file I/O error handling across saves/loads (not just shard deletion).

# todo:
# cleanup tables: shouldnt_exist_features should be dropped in previous steps.
# Regex not matching any column: len:10
# [('board_result_id', 'deal'), ('Club', 'event'), ('MasterPoints_[NESW]', 'players'), ('board_scoring_method', 'session'), ('club_session', 'session'), ('Round', 'session'), ('Table', 'session'), ('tb_count', 'session'), ('Lead', 'opening_lead'), ('HandRecordBoardScore', 'board_results')]


# Recommendations to generalize for multiple y_names
# Schema changes
# Add targets object keyed by y_name with:
# dtype (regression/classification), y_range, scaling (if needed), metrics.
# Store default_target and allow overriding at train/infer time.
# Persist per-target stats (mean/std, min/max) to inform scaling and clamps.
# Data/Tensors
# Allow multiple targets in df_to_scaled_tensors:
# Accept y_names: List[str], return y: torch.Tensor shape [N] or [N, T].
# Do not clip/scale targets globally; use per-target config from schema.
# Shards: store y as dict or a stacked tensor with target_names in shard metadata.
# Model heads
# Support multi-output regression/classification:
# Regression: final nn.Linear(last_dim, num_targets).
# Classification: optional heads per target (e.g., dict of nn.Linear/nn.Sequential) if heterogeneous tasks.
# Map outputs to column names via schema targets order.
# Loss and metrics
# Loss: compute per-target loss and aggregate (weighted average configurable in schema).
# Metrics: compute per-target (MAE/RMSE for regression; accuracy/AUC for classification) and log separately.
# Training API
# train_pct_ns_model_using_df/shards:
# Accept y_names: List[str] (≥1).
# Determine task_type per target from schema; build appropriate head(s).
# Save per-target best metrics/early stopping state.
# Early stopping: track the primary target (default from schema), allow custom selection.
# Inference API
# predict_regression_model → predict_model:
# Accept y_names or default to schema targets.
# Produce one prediction column per target (e.g., Target_Pred) with optional per-target clamp using targets[y]['y_range'].
# Append per-target error columns if actuals provided.
# Iterator/batching
# No change for X; expand y batching to handle [N, T] or dict keyed by y_name.
# Validation/consistency
# Extend validate_feature_consistency to also check:
# Targets present and dtypes match schema.
# Model output dimension matches number of requested targets (or head names).
# Feature importance
# If multi-output, compute and display importance per target (e.g., weight norms from per-target head or gradient-based approximations), with target selector.
# Charts/analysis
# Generalize analyze_prediction_results to accept target_col or iterate all y_names and emit per-target panels (calibration, error histogram, residuals).
# Config and defaults
# Introduce a small config block (defaults) for:
# Per-target y_range, loss weights, metrics, output activation (e.g., sigmoid for bounded targets).
# Keep backward compatibility: if single y_name, behave as now.
# Backward-compatible path
# If y_names length == 1:
# Build single-output head and keep existing column names.
# If y_names length > 1:
# Use multi-output head and produce one set of columns per target.
# These changes will let you train/predict multiple targets in one pass, or flexibly switch targets without modifying core code.


import polars as pl
import pathlib
from collections import defaultdict
from enum import Enum
import time
import json
import re
from tqdm import tqdm
from IPython.display import display
import numpy as np
import logging
from mlBridgeLib.logging_config import setup_logger
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from typing import Any, Optional, Dict, List, Tuple, Iterable, Callable, Union
import matplotlib.pyplot as plt
import seaborn as sns
import mlBridgeLib.mlBridgeLib as mlBridgeLib


logger = setup_logger(__name__)

def print_to_log_info(*args: Any) -> None:
    print_to_log(logging.INFO, *args)

def print_to_log_debug(*args: Any) -> None:
    print_to_log(logging.DEBUG, *args)

def print_to_log(level: int, *args: Any) -> None:
    logger.log(level, ' '.join(str(arg) for arg in args))


class features_enum(Enum):
    board_game_state = 0,
    deal_game_state = 1,
    event_game_state = 2,
    players_game_state = 3,
    session_game_state = 4,
    final_contract_game_state = 5,
    opening_lead_game_state = 6,
    dummy_game_state = 7,
    play_game_state = 8,
    board_results_game_state = 9,
    matchpoint_game_state = 10,
    rank_game_state = 11,


def concat_features(X_cont: torch.Tensor, X_cat: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Purpose:
        Concatenate continuous and categorical feature tensors into a single input tensor.

    Args:
        X_cont: Continuous features tensor [batch_size, num_continuous].
        X_cat: Optional categorical features tensor [batch_size, num_categorical].
               If None/empty, only continuous features are returned.

    Return:
        Concatenated tensor [batch_size, num_continuous + num_categorical].
    """
    if X_cat is None or X_cat.numel() == 0:
        return X_cont
    return torch.cat([X_cont, X_cat.float()], dim=1)


def iterate_batches(
    X_cont: torch.Tensor,
    X_cat: Optional[torch.Tensor],
    y: Optional[torch.Tensor],
    batch_size: int
):
    """
    Purpose:
        Yield mini-batches as (xb_cont, xb_cat, yb) to support models with optional categorical inputs.

    Args:
        X_cont: Continuous features [N, C] (float32).
        X_cat: Optional categorical features [N, K] (int64) or None.
        y: Optional targets [N] (float32 for regression, int64 for classification), or None for inference.
        batch_size: Rows per batch.

    Returns:
        Iterator yielding (xb_cont, xb_cat, yb):
          - xb_cont: torch.FloatTensor [B, C]
          - xb_cat:  torch.LongTensor [B, K] or None
          - yb:      torch.FloatTensor [B] or torch.LongTensor [B], or None
    """
    n = X_cont.shape[0]
    for i in range(0, n, batch_size):
        xb_cont = X_cont[i:i+batch_size]
        xb_cat = None if X_cat is None else X_cat[i:i+batch_size]
        yb = None if y is None else y[i:i+batch_size]
        yield xb_cont, xb_cat, yb


def setup_amp(device: str, use_amp: bool):
    """
    Purpose:
        Configure device, GradScaler, and autocast context for mixed precision training.

    Args:
        device: Preferred device string ('cuda' or 'cpu').
        use_amp: Whether to enable AMP when CUDA is available.

    Return:
        (device_t, scaler, autocast_fn)
    """
    device_t = torch.device(device if device != 'cpu' and torch.cuda.is_available() else 'cpu')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device_t.type == 'cuda')
    autocast_fn = lambda: torch.amp.autocast('cuda', enabled=use_amp and device_t.type == 'cuda')
    return device_t, scaler, autocast_fn


def train_one_epoch(
    model: torch.nn.Module,
    iterator,
    loss_fn,
    optimizer,
    device_t: torch.device,
    y_range: Optional[Tuple[float, float]],
    scaler,
    use_amp: bool,
    grad_clip: Optional[float] = 1.0
) -> Tuple[float, int]:
    model.train()
    total_loss, total_count = 0.0, 0
    for xb_cont, xb_cat, yb in iterator:
        xb_cont = xb_cont.to(device_t, non_blocking=True)
        xb_cat = xb_cat.to(device_t, non_blocking=True) if xb_cat is not None else None
        yb = yb.to(device_t, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp and device_t.type == 'cuda'):
            pred = model(xb_cont, xb_cat).squeeze(-1)
            if y_range is not None:
                pred = pred.clamp(*y_range)
            mask = torch.isfinite(pred) & torch.isfinite(yb)
            if not mask.any():
                continue
            loss = loss_fn(pred[mask], yb[mask])
        if use_amp and device_t.type == 'cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        total_loss += loss.item() * mask.sum().item()
        total_count += mask.sum().item()
    return total_loss, total_count


def eval_one_epoch(
    model: torch.nn.Module,
    iterator,
    loss_fn,
    device_t: torch.device,
    y_range: Optional[Tuple[float, float]]
) -> Tuple[float, int]:
    model.eval()
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for xb_cont, xb_cat, yb in iterator:
            xb_cont = xb_cont.to(device_t, non_blocking=True)
            xb_cat = xb_cat.to(device_t, non_blocking=True) if xb_cat is not None else None
            yb = yb.to(device_t, non_blocking=True)
            pred = model(xb_cont, xb_cat).squeeze(-1)
            if y_range is not None:
                pred = pred.clamp(*y_range)
            mask = torch.isfinite(pred) & torch.isfinite(yb)
            if not mask.any():
                continue
            loss = loss_fn(pred[mask], yb[mask])
            total_loss += loss.item() * mask.sum().item()
            total_count += mask.sum().item()
    return total_loss, total_count


def eval_one_epoch_with_stats(
    model: torch.nn.Module,
    iterator,
    loss_fn,
    device_t: torch.device,
    y_range: Optional[Tuple[float, float]]
) -> Tuple[float, int, Optional[float], Optional[float]]:
    """Enhanced evaluation that also returns prediction mean and std"""
    model.eval()
    total_loss, total_count = 0.0, 0
    all_preds = []
    
    with torch.no_grad():
        for xb_cont, xb_cat, yb in iterator:
            xb_cont = xb_cont.to(device_t, non_blocking=True)
            xb_cat = xb_cat.to(device_t, non_blocking=True) if xb_cat is not None else None
            yb = yb.to(device_t, non_blocking=True)
            pred = model(xb_cont, xb_cat).squeeze(-1)
            if y_range is not None:
                pred = pred.clamp(*y_range)
            mask = torch.isfinite(pred) & torch.isfinite(yb)
            if not mask.any():
                continue
            loss = loss_fn(pred[mask], yb[mask])
            total_loss += loss.item() * mask.sum().item()
            total_count += mask.sum().item()
            
            # Collect predictions for statistics
            all_preds.append(pred[mask].cpu())
    
    # Calculate prediction statistics
    pred_mean, pred_std = None, None
    if all_preds:
        all_preds_tensor = torch.cat(all_preds)
        if len(all_preds_tensor) > 0:
            pred_mean = float(all_preds_tensor.mean())
            pred_std = float(all_preds_tensor.std())
    
    return total_loss, total_count, pred_mean, pred_std


def eval_classification_with_stats(
    model: torch.nn.Module,
    iterator,
    loss_fn,
    device_t: torch.device
) -> Tuple[float, int, float, Optional[float], Optional[float]]:
    """Enhanced classification evaluation that returns accuracy and prediction statistics"""
    model.eval()
    total_loss, total_count, correct = 0.0, 0, 0
    all_probs = []
    
    with torch.no_grad():
        for xb, yb in iterator:
            xb = xb.to(device_t, non_blocking=True)
            yb = yb.to(device_t, non_blocking=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            
            # Calculate accuracy
            pred_classes = torch.argmax(logits, dim=1)
            correct += (pred_classes == yb).sum().item()
            
            # Collect prediction probabilities for statistics
            probs = torch.softmax(logits, dim=1)
            max_probs = torch.max(probs, dim=1)[0]  # Max probability for each prediction
            all_probs.append(max_probs.cpu())
            
            total_loss += loss.item() * len(yb)
            total_count += len(yb)
    
    accuracy = correct / max(1, total_count)
    
    # Calculate prediction confidence statistics
    pred_mean, pred_std = None, None
    if all_probs:
        all_probs_tensor = torch.cat(all_probs)
        if len(all_probs_tensor) > 0:
            pred_mean = float(all_probs_tensor.mean())  # Mean confidence
            pred_std = float(all_probs_tensor.std())    # Std of confidence
    
    return total_loss, total_count, accuracy, pred_mean, pred_std


def predict_batches(
    model: torch.nn.Module,
    iterator,
    device_t: torch.device,
    use_amp: bool,
    y_range: Optional[Tuple[float, float]] = None
) -> List[float]:
    model.eval()
    preds: List[float] = []
    with torch.no_grad():
        for xb_cont, xb_cat, _ in iterator:
            xb_cont = xb_cont.to(device_t, non_blocking=True)
            xb_cat = xb_cat.to(device_t, non_blocking=True) if xb_cat is not None else None
            with torch.amp.autocast('cuda', enabled=use_amp and device_t.type == 'cuda'):
                batch_preds = model(xb_cont, xb_cat).squeeze(-1)
                if y_range is not None:
                    batch_preds = batch_preds.clamp(*y_range)
            preds.extend(batch_preds.cpu().tolist())
    return preds


def load_shard_data(shard_file: pathlib.Path) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Purpose:
        Load a shard produced via torch.save with keys 'X_cont', 'X_cat' (optional), and 'y'.

    Behavior:
        - X_cont is converted to float32 and NaNs/±inf are cleaned (0.0, ±1e6).
        - X_cat is converted to int64 if present; remains None if absent/empty.
        - y dtype is preserved (float32 for regression, int64 for classification).
          If y is float64, it is downcast to float32.

    Args:
        shard_file: Path to shard .pt file.

    Returns:
        (X_cont, X_cat, y) where:
          - X_cont: torch.FloatTensor [N, C] or None
          - X_cat:  torch.LongTensor [N, K] or None
          - y:      torch.FloatTensor [N] or torch.LongTensor [N]
    """
    payload = torch.load(shard_file, map_location='cpu')
    X_cont = payload.get('X_cont', None)
    X_cat = payload.get('X_cat', None)
    y_val = payload.get('y')

    X_cont_t = torch.as_tensor(X_cont, dtype=torch.float32) if X_cont is not None else None
    # Keep categories as int64 for embeddings or indexing if used
    X_cat_t = torch.as_tensor(X_cat, dtype=torch.int64) if X_cat is not None else None

    # Leave y dtype as-is; regression should be float, classification should be long
    y_t = torch.as_tensor(y_val)
    if y_t.dtype == torch.float64:
        y_t = y_t.to(torch.float32)

    return X_cont_t, X_cat_t, y_t


def shard_batch_iterator(files: List[pathlib.Path], batch_size: int):
    """
    Purpose:
        Stream shards from disk and yield mini-batches using iterate_batches.

    Args:
        files: List of shard file paths.
        batch_size: Rows per batch to emit.

    Yields:
        3-tuples (xb_cont, xb_cat, yb) exactly matching iterate_batches output:
          - xb_cont: torch.FloatTensor [B, C]
          - xb_cat:  torch.LongTensor [B, K] or None
          - yb:      torch.FloatTensor [B] or torch.LongTensor [B]
    """
    for f in files:
        X_cont, X_cat, y = load_shard_data(f)
        for xb_cont, xb_cat, yb in iterate_batches(X_cont, X_cat, y, batch_size):
            yield xb_cont, xb_cat, yb


def resolve_model_path(saved_models_path: pathlib.Path, model_name: str) -> pathlib.Path:
    """Return fully-qualified model path from a models directory and filename."""
    return saved_models_path / f"{model_name}.pth"


def resolve_shards_path(saved_models_path: pathlib.Path, model_name: str, shard) -> pathlib.Path:
    """Return fully-qualified shards path from a models directory and filename."""
    return saved_models_path / f"{model_name}_shard_{shard:09d}.pt"


def resolve_schema_path(saved_models_path: pathlib.Path, model_name: str) -> pathlib.Path:
    """Return fully-qualified schema path from a models directory and filename."""
    return saved_models_path / f"{model_name}_schema.json"


def resolve_raw_path(saved_models_path: pathlib.Path, raw_prefix: str) -> pathlib.Path:
    """Return path for the raw shards manifest file (raw_prefix is dataset/shard set id)."""
    return saved_models_path / f"{raw_prefix}_raw.pt"


def resolve_raw_shard_files(saved_models_path: pathlib.Path, raw_prefix: str) -> List[pathlib.Path]:
    """
    Load the raw shards manifest and return the list of shard file paths.
    The manifest is saved at resolve_raw_path(...) and expected to contain a dict with 'shards'.
    """
    manifest_path = resolve_raw_path(saved_models_path, raw_prefix)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Raw manifest not found: {manifest_path}")
    manifest = torch.load(manifest_path, map_location='cpu')
    shard_strs = manifest.get('shards', [])
    # Backward compatibility: if manifest is missing or empty, fallback to glob
    if not shard_strs:
        return sorted(saved_models_path.glob(f"{raw_prefix}_raw_shard_*.pt"))
    return [pathlib.Path(p) for p in shard_strs]


def format_epoch_line(epoch_idx: int, epochs: int, train_loss: float, val_loss: Optional[float], epoch_samples: int, val_samples: int) -> str:
    """
    Purpose:
        Create a concise, human-readable summary for an epoch.

    Args:
        epoch_idx: Zero-based epoch index.
        epochs: Total epochs to run.
        train_loss: Mean training loss for the epoch.
        val_loss: Mean validation loss for the epoch (or None).
        epoch_samples: Number of weighted samples used for training mean loss.
        val_samples: Number of weighted samples used for validation mean loss.

    Return:
        A formatted string suitable for logging.
    """
    parts = [
        f"Epoch {epoch_idx+1}/{epochs}",
        f"train_loss={train_loss:.6f}",
        f"train_samples={epoch_samples}"
    ]
    if val_loss is not None:
        parts.append(f"val_loss={val_loss:.6f}")
        parts.append(f"val_samples={val_samples}")
    return ", ".join(parts)


def format_epoch_line_with_stats(
    epoch_idx: int, 
    epochs: int, 
    train_loss: float, 
    val_loss: Optional[float], 
    epoch_samples: int, 
    val_samples: int,
    val_pred_mean: Optional[float] = None,
    val_pred_std: Optional[float] = None
) -> str:
    """
    Enhanced epoch formatting that includes validation prediction statistics.
    
    Args:
        epoch_idx: Zero-based epoch index.
        epochs: Total epochs to run.
        train_loss: Mean training loss for the epoch.
        val_loss: Mean validation loss for the epoch (or None).
        epoch_samples: Number of weighted samples used for training mean loss.
        val_samples: Number of weighted samples used for validation mean loss.
        val_pred_mean: Mean of validation predictions (or None).
        val_pred_std: Standard deviation of validation predictions (or None).
    
    Return:
        A formatted string suitable for logging with prediction statistics.
    """
    parts = [
        f"Epoch {epoch_idx+1}/{epochs}",
        f"train_loss={train_loss:.6f}",
        f"train_samples={epoch_samples}"
    ]
    if val_loss is not None:
        parts.append(f"val_loss={val_loss:.6f}")
        parts.append(f"val_samples={val_samples}")
        
        # Add prediction statistics if available
        if val_pred_mean is not None and val_pred_std is not None:
            parts.append(f"val_pred_mean={val_pred_mean:.4f}")
            parts.append(f"val_pred_std={val_pred_std:.4f}")
    
    return ", ".join(parts)


def format_epoch_line_classification(
    epoch_idx: int, 
    epochs: int, 
    train_loss: float, 
    val_loss: Optional[float], 
    epoch_samples: int, 
    val_samples: int,
    val_accuracy: Optional[float] = None,
    val_conf_mean: Optional[float] = None,
    val_conf_std: Optional[float] = None
) -> str:
    """
    Classification-specific epoch formatting with accuracy and confidence statistics.
    
    Args:
        epoch_idx: Zero-based epoch index.
        epochs: Total epochs to run.
        train_loss: Mean training loss for the epoch.
        val_loss: Mean validation loss for the epoch (or None).
        epoch_samples: Number of weighted samples used for training mean loss.
        val_samples: Number of weighted samples used for validation mean loss.
        val_accuracy: Validation accuracy (or None).
        val_conf_mean: Mean prediction confidence (or None).
        val_conf_std: Standard deviation of prediction confidence (or None).
    
    Return:
        A formatted string suitable for logging with classification statistics.
    """
    parts = [
        f"Epoch {epoch_idx+1}/{epochs}",
        f"train_loss={train_loss:.6f}",
        f"train_samples={epoch_samples}"
    ]
    if val_loss is not None:
        parts.append(f"val_loss={val_loss:.6f}")
        parts.append(f"val_samples={val_samples}")
        
        # Add classification-specific statistics
        if val_accuracy is not None:
            parts.append(f"val_acc={val_accuracy:.4f}")
        if val_conf_mean is not None and val_conf_std is not None:
            parts.append(f"val_conf_mean={val_conf_mean:.4f}")
            parts.append(f"val_conf_std={val_conf_std:.4f}")
    
    return ", ".join(parts)


def format_epoch_line_with_stats(
    epoch_idx: int, 
    epochs: int, 
    train_loss: float, 
    val_loss: Optional[float], 
    epoch_samples: int, 
    val_samples: int,
    val_pred_mean: Optional[float] = None,
    val_pred_std: Optional[float] = None
) -> str:
    """
    Enhanced epoch formatting that includes validation prediction statistics.
    
    Args:
        epoch_idx: Zero-based epoch index.
        epochs: Total epochs to run.
        train_loss: Mean training loss for the epoch.
        val_loss: Mean validation loss for the epoch (or None).
        epoch_samples: Number of weighted samples used for training mean loss.
        val_samples: Number of weighted samples used for validation mean loss.
        val_pred_mean: Mean of validation predictions (or None).
        val_pred_std: Standard deviation of validation predictions (or None).
    
    Return:
        A formatted string suitable for logging with prediction statistics.
    """
    parts = [
        f"Epoch {epoch_idx+1}/{epochs}",
        f"train_loss={train_loss:.6f}",
        f"train_samples={epoch_samples}"
    ]
    if val_loss is not None:
        parts.append(f"val_loss={val_loss:.6f}")
        parts.append(f"val_samples={val_samples}")
        
        # Add prediction statistics if available
        if val_pred_mean is not None and val_pred_std is not None:
            parts.append(f"val_pred_mean={val_pred_mean:.4f}")
            parts.append(f"val_pred_std={val_pred_std:.4f}")
    
    return ", ".join(parts)


def format_epoch_line_classification(
    epoch_idx: int, 
    epochs: int, 
    train_loss: float, 
    val_loss: Optional[float], 
    epoch_samples: int, 
    val_samples: int,
    val_accuracy: Optional[float] = None,
    val_conf_mean: Optional[float] = None,
    val_conf_std: Optional[float] = None
) -> str:
    """
    Classification-specific epoch formatting with accuracy and confidence statistics.
    
    Args:
        epoch_idx: Zero-based epoch index.
        epochs: Total epochs to run.
        train_loss: Mean training loss for the epoch.
        val_loss: Mean validation loss for the epoch (or None).
        epoch_samples: Number of weighted samples used for training mean loss.
        val_samples: Number of weighted samples used for validation mean loss.
        val_accuracy: Validation accuracy (or None).
        val_conf_mean: Mean prediction confidence (or None).
        val_conf_std: Standard deviation of prediction confidence (or None).
    
    Return:
        A formatted string suitable for logging with classification statistics.
    """
    parts = [
        f"Epoch {epoch_idx+1}/{epochs}",
        f"train_loss={train_loss:.6f}",
        f"train_samples={epoch_samples}"
    ]
    if val_loss is not None:
        parts.append(f"val_loss={val_loss:.6f}")
        parts.append(f"val_samples={val_samples}")
        
        # Add classification-specific statistics
        if val_accuracy is not None:
            parts.append(f"val_acc={val_accuracy:.4f}")
        if val_conf_mean is not None and val_conf_std is not None:
            parts.append(f"val_conf_mean={val_conf_mean:.4f}")
            parts.append(f"val_conf_std={val_conf_std:.4f}")
    
    return ", ".join(parts)


def make_training_stats(
    total_epochs: int,
    best_val_loss: Optional[float],
    training_time: Optional[float],
    train_samples: int,
    val_samples: int,
    feature_cols: List[str],
    y_name: str,
    input_dim: int,
    device: torch.device,
    use_amp: bool
) -> Dict[str, Any]:
    """
    Purpose:
        Assemble a dictionary of training metadata and summary statistics.

    Args:
        total_epochs: Number of epochs actually completed.
        best_val_loss: Lowest validation loss observed.
        training_time: Total wall-clock training time in seconds.
        train_samples: Weighted sample count for training loss.
        val_samples: Weighted sample count for validation loss.
        feature_cols: Ordered list of features used for training.
        y_name: Target column name.
        input_dim: Model input dimension (continuous + embeddings where applicable).
        device: Torch device string used.
        use_amp: Whether AMP was enabled.

    Return:
        Dict[str, Any] containing the above fields for downstream use.
    """
    return {
        'total_epochs': total_epochs,
        'best_val_loss': best_val_loss,
        'training_time': training_time,
        'train_samples': train_samples,
        'val_samples': val_samples,
        'feature_cols': feature_cols,
        'y_name': y_name,
        'input_dim': input_dim,
        'device': str(device),
        'use_amp': use_amp,
    }


def build_regression_mlp(input_dim: int, layers: List[int], dropout: float = 0.2) -> torch.nn.Module:
    class MLP(torch.nn.Module):
        def __init__(self, dim_in: int, widths: List[int], dropout: float):
            super().__init__()
            blocks = []
            prev = dim_in
            for w in widths:
                blocks.append(torch.nn.Linear(prev, w))
                blocks.append(torch.nn.ReLU())
                if dropout and dropout > 0:
                    blocks.append(torch.nn.Dropout(dropout))
                prev = w
            blocks.append(torch.nn.Linear(prev, 1))
            self.layers = torch.nn.Sequential(*blocks)
        def forward(self, x):
            return self.layers(x).squeeze(-1)
    return MLP(input_dim, layers, dropout)


def build_classifier_mlp(input_dim: int, num_classes: int, layers: List[int], dropout: float = 0.2) -> torch.nn.Module:
    class Classifier(torch.nn.Module):
        def __init__(self, dim_in: int, widths: List[int], dropout: float, num_classes: int):
            super().__init__()
            blocks = []
            prev = dim_in
            for w in widths:
                blocks.append(torch.nn.Linear(prev, w))
                blocks.append(torch.nn.ReLU())
                if dropout and dropout > 0:
                    blocks.append(torch.nn.Dropout(dropout))
                prev = w
            blocks.append(torch.nn.Linear(prev, num_classes))
            self.layers = torch.nn.Sequential(*blocks)
        def forward(self, x):
            return self.layers(x)
    return Classifier(input_dim, layers, dropout, num_classes)


def df_to_tensors_regression(df: pl.DataFrame, y_name: str, feature_cols: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    X = df.select(feature_cols).to_numpy()
    y = df.select([y_name]).to_numpy().reshape(-1)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def df_to_tensors_classification(df: pl.DataFrame, y_name: str, feature_cols: List[str], class_to_idx: Dict[Any, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    X = df.select(feature_cols).to_numpy()
    y_raw = df.select([y_name]).to_numpy().reshape(-1)
    y_idx = np.vectorize(lambda v: class_to_idx[v])(y_raw)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y_idx, dtype=torch.long)


class MLP(torch.nn.Module):
    """Multi-Layer Perceptron for bridge percentage prediction with optional embeddings for categorical features.

    Shared between training and inference functions to ensure consistency.
    """
    def __init__(self, input_dim: int, cat_dims: Optional[List[Tuple[int, int]]] = None, layer_sizes: List[int] = [1024, 512, 256, 128], dropout: float = 0.1):
        super().__init__()

        self.embeddings = torch.nn.ModuleList()
        embedding_output_dim = 0
        if cat_dims:
            for c, emb_dim in cat_dims:
                self.embeddings.append(torch.nn.Embedding(c, emb_dim))
                embedding_output_dim += emb_dim

        # Adjust input_dim for the linear layers: original continuous features + embedding outputs
        total_input_dim = input_dim + embedding_output_dim

        layers = []
        prev_size = total_input_dim

        for size in layer_sizes:
            layers.extend([
                torch.nn.Linear(prev_size, size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
            ])
            prev_size = size

        # Output layer (no activation - raw logits for regression)
        layers.append(torch.nn.Linear(prev_size, 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x_cont, x_cat=None):
        # Process categorical features through embeddings if they exist
        if self.embeddings and x_cat is not None:
            embedded_features = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.embeddings)]
            x_cat_embedded = torch.cat(embedded_features, 1)
            # Concatenate continuous and embedded categorical features
            x = torch.cat([x_cont, x_cat_embedded], 1)
        else:
            x = x_cont # No categorical features or no embeddings defined

        return self.layers(x).squeeze(-1)  # Remove last dimension for scalar output


def compute_scaling_parameters(df: pl.DataFrame, apply_scaling: bool = True, verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Purpose:
        Compute per-feature (mean, std) for all numeric features.
        Handles multiple data types: Float, Int, Boolean, Date.

    Args:
        df: Polars DataFrame containing numerical feature columns (Float/Int/Boolean/Date).
        apply_scaling: Whether to compute scaling parameters.
        verbose: Whether to print progress information.

    Return:
        Dict mapping feature name -> {"mean": float, "std": float}.
    """
    scaling_params: Dict[str, Dict[str, float]] = {}

    if not apply_scaling:
        if verbose:
            print("❌ Scaling disabled - returning empty scaling parameters")
        return scaling_params

    numeric_cols = list(df.columns)
    if verbose:
        print(f"🔢 Computing scaling parameters for {len(numeric_cols)} numeric features...")

    for col in tqdm(numeric_cols, desc="scaling-features", leave=False):
        try:
            series = df[col]
            dt = series.dtype

            # Convert to float for scaling computation using Series ops only
            if dt == pl.Boolean:
                # Boolean: convert to 0/1
                numeric_series = series.cast(pl.Int8).cast(pl.Float64)
            elif dt == pl.Date:
                # Date: underlying logical type is days since epoch (Int32)
                numeric_series = series.cast(pl.Int32).cast(pl.Float64)
            elif dt == pl.Datetime:
                # Datetime: cast to Int64 (ns since epoch) then to float
                numeric_series = series.cast(pl.Int64).cast(pl.Float64)
            else:
                # Float/Int and other numeric-like types: direct conversion
                numeric_series = series.cast(pl.Float64)

            mean_val = numeric_series.mean()
            std_val = numeric_series.std()
            mean = 0.0 if (mean_val is None or not np.isfinite(mean_val)) else float(mean_val)
            std = float(std_val) if (std_val is not None and np.isfinite(std_val) and float(std_val) > 0.0) else 1.0

            scaling_params[col] = {"mean": mean, "std": std}
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to compute scaling for {col}: {e}")
            continue

    if verbose:
        print(f"✅ Computed scaling parameters for {len(scaling_params)} features")

    return scaling_params


def apply_feature_scaling(
    data: np.ndarray,
    feature_names: List[str],
    scaling_params: Dict[str, Dict[str, float]],
    apply_scaling: bool = True
) -> np.ndarray:
    """
    Apply per-feature standardization to a numpy array given scaling parameters.
    """
    if not apply_scaling or not scaling_params:
        return data

    scaled_data = data.copy()
    for idx, feature in enumerate(feature_names):
        params = scaling_params.get(feature)
        if params is None:
            continue
        mean = params.get("mean", 0.0)
        std = params.get("std", 1.0) or 1.0
        scaled_data[:, idx] = (scaled_data[:, idx] - mean) / std
    return scaled_data


def validate_dataframe_dtypes(
    df: pl.DataFrame,
    expected_dtypes: Dict[str, str],
    context: str = "DataFrame",
    strict: bool = True
) -> pl.DataFrame:
    """
    Validate that DataFrame columns have the expected dtypes.
    
    Args:
        df: Input DataFrame to validate
        expected_dtypes: Dict mapping column names to expected dtype strings
        context: Description for error messages (e.g., "training", "inference")
        strict: If True, raise errors on mismatches. If False, attempt conversion.
        
    Returns:
        DataFrame with validated/converted dtypes
        
    Raises:
        ValueError: If strict=True and dtypes don't match
    """
    df_schema = dict(df.schema)
    mismatches = []
    conversions_needed = []
    
    # Treat String/Utf8/Categorical as equivalent since they're processed identically
    string_like_types = {'String', 'Utf8', 'Categorical'}
    
    def normalize_dtype_str(dtype_str: str) -> str:
        """Normalize dtype string by stripping categorical ordering specs."""
        # Strip out ordering='physical' or ordering='lexical' from Categorical types
        if dtype_str.startswith('Categorical'):
            return 'Categorical'
        return dtype_str
    
    for col, expected_dtype_str in expected_dtypes.items():
        if col not in df_schema:
            continue  # Missing columns handled elsewhere
            
        actual_dtype = df_schema[col]
        actual_dtype_str = str(actual_dtype)
        
        # Normalize both to handle categorical ordering variations
        expected_normalized = normalize_dtype_str(expected_dtype_str)
        actual_normalized = normalize_dtype_str(actual_dtype_str)
        
        # Allow String/Utf8/Categorical to be used interchangeably
        if expected_normalized in string_like_types and actual_normalized in string_like_types:
            continue  # These are equivalent for our purposes
        
        if actual_normalized != expected_normalized:
            mismatches.append(f"  {col}: expected {expected_dtype_str}, got {actual_dtype_str}")
            conversions_needed.append((col, expected_dtype_str, actual_dtype_str))
    
    if mismatches:
        error_msg = f"Dtype mismatches in {context}:\n" + "\n".join(mismatches)
        
        if strict:
            raise ValueError(error_msg)
        else:
            print(f"⚠️  {error_msg}")
            print(f"🔧 Attempting dtype conversions...")
            
            # Attempt conversions
            df_converted = df
            for col, expected_dtype_str, actual_dtype_str in conversions_needed:
                try:
                    # Map string dtype names to Polars dtypes
                    dtype_map = {
                        'Float32': pl.Float32,
                        'Float64': pl.Float64,
                        'Int8': pl.Int8,
                        'Int16': pl.Int16,
                        'Int32': pl.Int32,
                        'Int64': pl.Int64,
                        'UInt8': pl.UInt8,
                        'UInt16': pl.UInt16,
                        'UInt32': pl.UInt32,
                        'UInt64': pl.UInt64,
                        'Boolean': pl.Boolean,
                        'String': pl.String,
                        'Utf8': pl.Utf8,
                        'Categorical': pl.Categorical,
                        'Date': pl.Date,
                        'Datetime': pl.Datetime,
                    }
                    
                    target_dtype = dtype_map.get(expected_dtype_str)
                    if target_dtype:
                        df_converted = df_converted.with_columns(
                            pl.col(col).cast(target_dtype)
                        )
                        print(f"  ✅ Converted {col}: {actual_dtype_str} → {expected_dtype_str}")
                    else:
                        print(f"  ❌ Unknown target dtype {expected_dtype_str} for {col}")
                        
                except Exception as e:
                    print(f"  ❌ Failed to convert {col}: {e}")
                    if strict:
                        raise ValueError(f"Failed to convert {col} from {actual_dtype_str} to {expected_dtype_str}: {e}")
            
            return df_converted
    
    return df


def validate_training_dataframe_dtypes(
    df: pl.DataFrame,
    feature_columns: List[str],
    target_column: str,
    verbose: bool = False
) -> pl.DataFrame:
    """
    Validate that training DataFrame has appropriate dtypes for ML training.
    
    Args:
        df: Training DataFrame
        feature_columns: List of feature column names
        target_column: Name of target column
        verbose: Print validation details
        
    Returns:
        DataFrame with validated dtypes
        
    Raises:
        ValueError: If critical dtype issues are found
    """
    if verbose:
        print(f"🔍 Validating dtypes for {len(feature_columns)} features + 1 target...")
    
    df_schema = dict(df.schema)
    issues = []
    
    # Check feature columns
    for col in feature_columns:
        if col not in df_schema:
            issues.append(f"Missing feature column: {col}")
            continue
            
        dtype = df_schema[col]
        dtype_str = str(dtype)
        
        # Features should be numeric, boolean, categorical, or string (for categoricals)
        if not any(t in dtype_str for t in ['Float', 'Int', 'UInt', 'Boolean', 'Categorical', 'String', 'Utf8', 'Date']):
            issues.append(f"Feature {col} has unsupported dtype: {dtype_str}")
    
    # Check target column
    if target_column not in df_schema:
        issues.append(f"Missing target column: {target_column}")
    else:
        target_dtype_str = str(df_schema[target_column])
        if verbose:
            print(f"  Target column '{target_column}': {target_dtype_str}")
    
    if issues:
        raise ValueError(f"Training DataFrame validation failed:\n" + "\n".join(f"  - {issue}" for issue in issues))
    
    if verbose:
        print(f"✅ Training DataFrame dtype validation passed")
    
    return df


def validate_inference_dataframe_dtypes(
    df: pl.DataFrame,
    schema: Dict[str, Any],
    strict: bool = True,
    verbose: bool = False
) -> pl.DataFrame:
    """
    Validate that inference DataFrame dtypes match the training schema.
    
    Args:
        df: Inference DataFrame
        schema: Training schema with feature_dtypes
        strict: If True, raise errors on mismatches and missing columns
        verbose: Print validation details
        
    Returns:
        DataFrame with validated dtypes
        
    Raises:
        ValueError: If required columns are missing or dtypes don't match (strict mode)
    """
    feature_dtypes = schema.get('feature_dtypes', {})
    if not feature_dtypes:
        if verbose:
            print("[WARNING] No feature_dtypes in schema, skipping dtype validation")
        return df
    
    if verbose:
        print(f"[DEBUG] Validating inference dtypes against schema (strict={strict})...")
    
    # Check for missing columns
    df_columns = set(df.columns)
    schema_columns = set(feature_dtypes.keys())
    missing_in_df = schema_columns - df_columns
    extra_in_df = df_columns - schema_columns
    
    if verbose:
        print(f"  Schema expects {len(schema_columns)} feature columns")
        print(f"  DataFrame has {len(df_columns)} columns")
        if missing_in_df:
            print(f"  ERROR: Missing columns: {len(missing_in_df)}")
        if extra_in_df:
            print(f"  WARNING: Extra columns: {len(extra_in_df)}")
    
    # Handle missing columns
    if missing_in_df:
        missing_list = sorted(list(missing_in_df))
        error_msg = f"Required feature columns missing from inference DataFrame:\n"
        error_msg += "\n".join(f"  - {col} (expected dtype: {feature_dtypes[col]})" for col in missing_list[:10])
        if len(missing_list) > 10:
            error_msg += f"\n  ... and {len(missing_list) - 10} more columns"
        
        if strict:
            raise ValueError(error_msg)
        else:
            print(f"⚠️  {error_msg}")
            print("🔧 Continuing with available columns only...")
    
    # Validate dtypes for available columns
    common_columns = df_columns & schema_columns
    if verbose:
        print(f"  Validating dtypes for {len(common_columns)} available columns")
    
    expected_dtypes = {col: feature_dtypes[col] for col in common_columns}
    
    # Validate dtypes
    validated_df = validate_dataframe_dtypes(
        df, expected_dtypes, context="inference data", strict=strict
    )
    
    if verbose:
        print(f"[OK] Inference DataFrame validation completed")
        if missing_in_df and not strict:
            print(f"  [WARNING] Note: {len(missing_in_df)} columns were missing but validation continued")
    
    return validated_df


# gpt-5 says: Shared helpers to deduplicate tensor construction across DF/shards/inference.
def build_continuous_tensor(
    df: pl.DataFrame,
    numerical_feature_cols: List[str],
    scaling_params: Dict[str, Dict[str, float]],
    apply_scaling: bool,
    scaling_mode: str = 'auto',
) -> torch.Tensor:
    """Construct X_cont from DataFrame with consistent NaN/Inf handling and scaling.
    
    Handles multiple data types:
    - Float/Int: Direct conversion
    - Boolean: Convert to 0/1
    - Date: Convert to Unix timestamp (days since epoch)
    """
    if not numerical_feature_cols:
        return torch.empty((len(df), 0), dtype=torch.float32)
    
    # Simplified approach: let Polars handle the conversion automatically
    processed_cols = []
    for col in numerical_feature_cols:
        try:
            # Use select with cast to ensure we get a numpy-convertible result
            col_data = df.select(pl.col(col).cast(pl.Float32)).to_numpy().flatten()
            processed_cols.append(col_data)
        except:
            # Fallback: handle special data types manually
            col_series = df.select(col).to_series()
            dtype_str = str(col_series.dtype)
            
            if 'Boolean' in dtype_str:
                # Convert Boolean to 0/1
                processed_cols.append(col_series.cast(pl.Float32).to_numpy())
            elif 'Date' in dtype_str:
                # Convert Date to timestamp seconds
                timestamp_data = df.select(pl.col(col).dt.timestamp('s').cast(pl.Float32)).to_numpy().flatten()
                processed_cols.append(timestamp_data)
            else:
                # Try direct conversion
                processed_cols.append(col_series.cast(pl.Float32).to_numpy())
    
    # Stack all columns
    X_num = np.column_stack(processed_cols) if len(processed_cols) > 1 else processed_cols[0].reshape(-1, 1)
    X_num = np.nan_to_num(X_num, nan=0.0, posinf=1e6, neginf=-1e6)
    
    do_scale = (scaling_mode == 'on') or (scaling_mode == 'auto' and apply_scaling)
    if do_scale and scaling_params:
        X_num = apply_feature_scaling(X_num, numerical_feature_cols, scaling_params, apply_scaling=True)
    return torch.tensor(X_num, dtype=torch.float32)


def validate_categorical_schemas(
    df: pl.DataFrame,
    categorical_feature_cols: List[str],
    category_mappings: Dict[str, Dict[Any, int]]
) -> None:
    """
    Validate that inference data categorical schemas match training schemas exactly.
    
    This ensures that:
    1. All categorical columns exist in the inference data
    2. All categories in inference data were seen during training
    3. Training and inference use the same categorical value sets
    
    Raises ValueError if any mismatches are found.
    """
    try:
        # Import training schemas for comparison
        import sys
        import pathlib
        sys.path.append(str(pathlib.Path(__file__).parent))
        from mlBridgeLib import CATEGORICAL_SCHEMAS
        
        for col in categorical_feature_cols:
            if col not in df.columns:
                raise ValueError(f"Categorical column '{col}' not found in inference data")
            
            # Get training schema for this column
            if col not in CATEGORICAL_SCHEMAS:
                raise ValueError(f"No training schema found for categorical column '{col}'")
            
            training_categories = set(CATEGORICAL_SCHEMAS[col].categories)
            schema_categories = set(category_mappings.get(col, {}).keys())
            
            # Check schema matches training
            if schema_categories != training_categories:
                missing_in_schema = training_categories - schema_categories
                extra_in_schema = schema_categories - training_categories
                error_msg = f"Schema mismatch for column '{col}':"
                if missing_in_schema:
                    error_msg += f" Missing from schema: {sorted(missing_in_schema)}."
                if extra_in_schema:
                    error_msg += f" Extra in schema: {sorted(extra_in_schema)}."
                raise ValueError(error_msg)
            
            # Get inference data categories (excluding nulls)
            col_data = df.select(col).to_series()
            
            # Get unique values, excluding nulls
            unique_values = col_data.unique().to_list()
            inference_categories = set(str(v) for v in unique_values if v is not None)
            
            # Check inference data only contains training categories
            unknown_categories = inference_categories - training_categories
            if unknown_categories:
                raise ValueError(
                    f"Unknown categories in inference data for column '{col}': {sorted(unknown_categories)}. "
                    f"Training categories: {sorted(training_categories)}. "
                    f"This indicates the inference data contains values not seen during training."
                )
                
    except ImportError:
        # If we can't import training schemas, skip validation with warning
        print("Warning: Could not import training categorical schemas for validation")


def build_categorical_tensor(
    df: pl.DataFrame,
    categorical_feature_cols: List[str],
    category_mappings: Dict[str, Dict[Any, int]],
    cat_feature_info: Dict[str, int]
) -> Optional[torch.Tensor]:
    """Construct X_cat (indices) using fast Polars replace_strict with unknown handling.
    
    Handles multiple categorical data types:
    - String: Map to indices using category_mappings
    - Categorical: Map to indices using category_mappings
    """
    if not categorical_feature_cols:
        return None
    
    # Validate that training and inference categorical schemas match (once per session)
    if not hasattr(validate_categorical_schemas, '_validated_models'):
        validate_categorical_schemas._validated_models = set()
    
    # Create a unique key for this model's categorical schema
    schema_key = tuple(sorted(categorical_feature_cols)) + tuple(sorted(
        (col, tuple(sorted(mapping.items()))) 
        for col, mapping in category_mappings.items()
    ))
    
    if schema_key not in validate_categorical_schemas._validated_models:
        validate_categorical_schemas(df, categorical_feature_cols, category_mappings)
        validate_categorical_schemas._validated_models.add(schema_key)
    cat_arrays = []
    for col in categorical_feature_cols:
        mapping = category_mappings.get(col, {})
        if not mapping:
            raise ValueError(f"No categorical mapping found for column '{col}' in schema")
        
        # Ensure we get a proper Series, not an expression
        col_data = df.select(col).to_series()
        dtype_str = str(col_data.dtype)
        
        # Handle String columns by converting to string first, then mapping
        if dtype_str in ['String', 'Utf8']:
            # Convert nulls to string representation for checking
            col_data = col_data.fill_null("__NULL__")
        
        # Check for unknown categories before mapping
        unique_values = set(str(v) for v in col_data.unique().to_list() if v is not None)
        known_categories = set(mapping.keys())
        unknown_categories = unique_values - known_categories
        
        if unknown_categories:
            raise ValueError(
                f"Unknown categories found in column '{col}': {sorted(unknown_categories)}. "
                f"Known categories from training: {sorted(known_categories)}. "
                f"This indicates a mismatch between training and inference data."
            )
        
        # Map categories to indices (no default since we've verified all are known)
        mapped = (
            col_data
            .replace_strict(mapping, return_dtype=pl.Int64)
            .to_numpy()
            .astype(np.int64)
        )
        cat_arrays.append(mapped)
    if not cat_arrays:
        return None
    
    result = torch.tensor(np.stack(cat_arrays, axis=1), dtype=torch.int64)
    return result


def cat_dims_from_schema(schema: Dict[str, Any]) -> Optional[List[Tuple[int, int]]]:
    """Return list of (cardinality, emb_dim) from schema or None if no categoricals."""
    cat_feature_info = schema.get('cat_feature_info', {})
    if not cat_feature_info:
        return None
    # Reserve an extra index for unknowns (schema maps unknown to index==cardinality)
    return [(cardinality + 1, min(50, cardinality // 2)) for cardinality in cat_feature_info.values()]


def expected_input_dim_from_schema(schema: Dict[str, Any]) -> int:
    """Compute expected MLP input dim = num_continuous + sum(embedding_dims)."""
    numerical_feature_cols = schema.get('numerical_feature_cols', [])
    cat_feature_info = schema.get('cat_feature_info', {})
    # Embeddings reserve +1 slot for unknown index
    embedding_output_dim = sum(min(50, (cardinality // 2)) for cardinality in cat_feature_info.values())
    return len(numerical_feature_cols) + embedding_output_dim


def build_model_from_schema(schema: Dict[str, Any], input_dim: int, device: torch.device) -> torch.nn.Module:
    """Construct MLP on device using schema settings and cat dims (if any)."""
    mlp_layers = schema.get('mlp_layers', [512, 256, 128])
    dropout = schema.get('mlp_dropout', 0.05)
    cat_dims = cat_dims_from_schema(schema)
    model = MLP(input_dim=input_dim, cat_dims=cat_dims, layer_sizes=mlp_layers, dropout=dropout)
    return model.to(device)


def make_iter_fn(Xc: torch.Tensor, Xk: Optional[torch.Tensor], y: Optional[torch.Tensor], bs: int):
    """Factory to create a new iterator over in-memory tensors each epoch."""
    def _fn():
        return iterate_batches(Xc, Xk, y, bs)
    return _fn


def make_shard_iter_fn(files: List[pathlib.Path], bs: int, loader_fn):
    """Factory for shard-based iterators; calls loader_fn per file and yields mini-batches."""
    def _fn():
        for f in files:
            Xc, Xk, y = loader_fn(f)
            for xb in iterate_batches(Xc, Xk, y, bs):
                yield xb
    return _fn


def load_schema(base_path: pathlib.Path, filename: str) -> Dict[str, Any]:
    schema_path = resolve_schema_path(base_path, filename)
    return json.load(open(schema_path))


def save_schema(base_path: pathlib.Path, schema: Dict[str, Any], filename) -> pathlib.Path:
    schema_path = resolve_schema_path(base_path, filename)
    json.dump(schema, open(schema_path, 'w'))
    return schema_path


def safe_remove_file(path: pathlib.Path, retries: int = 2, delay: float = 0.1) -> bool:
    for attempt in range(retries + 1):
        try:
            path.unlink()
            return True
        except Exception:
            if attempt == retries:
                return False
            time.sleep(delay)
    return False


def safe_torch_save(obj: Any, path: pathlib.Path, retries: int = 1) -> None:
    for attempt in range(retries + 1):
        try:
            torch.save(obj, path)
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(0.1)


def safe_torch_load(path: pathlib.Path, map_location: str = 'cpu', retries: int = 1) -> Any:
    last_exc: Exception = None
    for attempt in range(retries + 1):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except Exception as e:
            last_exc = e
            if attempt == retries:
                break
            time.sleep(0.1)
    raise RuntimeError(f"Failed to load torch object from {path}: {last_exc}")


# gpt-5 says: Add schema version and y_range; persist run metadata (hyperparams, date, git SHA) for experiment tracking.
def generate_and_save_schema_core(
    df: pl.DataFrame,
    saved_models_path: pathlib.Path,
    model_name: str,
    y_name: str,
    layers: Optional[List[int]] = [1024, 512, 256, 128],
    dropout: float = 0.1,
    apply_scaling_parameters: bool = True,
    verbose: bool = False,
    y_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    schema_version: str = "1.1",
    blacklist_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generates and saves the model schema to schema.json.
    This function acts as the single source of truth for schema definition.
    Includes columns, scaling params, and categorical mappings for consistency
    across df-based and shard-based pipelines.
    
    Args:
        blacklist_patterns: Optional list of regex patterns to exclude columns.
                          Default patterns exclude name-related columns.
                          Examples: [r'.*name.*', r'^temp_.*', r'.*_id$']
    """
    schema_file = resolve_schema_path(saved_models_path, model_name)
    saved_models_path.mkdir(parents=True, exist_ok=True)

    # Determine columns and dtypes from df.schema (Polars): include all data types, blacklist only unwanted columns
    schema_map: Dict[str, Any] = dict(df.schema)
    cols = list(schema_map.keys())
    dtypes = schema_map  # map: column -> pl.DataType
    
    # Validate training DataFrame dtypes before processing
    all_feature_cols_preliminary = [c for c in cols if c != y_name]
    df = validate_training_dataframe_dtypes(
        df, all_feature_cols_preliminary, y_name, verbose=verbose
    )
    
    # Helpers to check Polars dtype kinds
    def _is_float_dtype(dt: Any) -> bool:
        return dt in (pl.Float32, pl.Float64)
    def _is_integer_dtype(dt: Any) -> bool:
        return dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    def _is_boolean_dtype(dt: Any) -> bool:
        return dt == pl.Boolean
    def _is_string_dtype(dt: Any) -> bool:
        # pl.Utf8 is the canonical string dtype; pl.String may alias
        return dt in (pl.Utf8, getattr(pl, 'String', pl.Utf8))
    def _is_date_dtype(dt: Any) -> bool:
        return dt in (pl.Date, pl.Datetime)
    def _is_categorical_dtype(dt: Any) -> bool:
        return dt == pl.Categorical
    
    # Blacklist patterns for columns to exclude (supports regex patterns)
    if blacklist_patterns is None:
        blacklist_patterns = [
            r'.*name.*',           # Any column containing 'name' (case-insensitive)
            r'player_name_[nesw]', # Specific player name columns
            r'declarer_name',      # Declarer name column
            # Add more regex patterns as needed:
            # r'^id$',             # Exact match for 'id' column
            # r'.*_id$',           # Any column ending with '_id'
            # r'^temp_.*',         # Any column starting with 'temp_'
        ]
    
    blacklisted_cols = []
    for col in cols:
        for pattern in blacklist_patterns:
            if re.match(pattern, col, re.IGNORECASE):
                blacklisted_cols.append(col)
                break  # Stop checking other patterns for this column
    
    if verbose and blacklisted_cols:
        print(f"🚫 Blacklisted columns ({len(blacklisted_cols)}): {blacklisted_cols}")
        print(f"🚫 Blacklist patterns used: {blacklist_patterns}")
    
    # Include ALL columns except target and blacklisted ones
    all_feature_cols = [c for c in cols if c != y_name and c not in blacklisted_cols]
    
    # Categorize by data type for processing
    feature_float_cols = [c for c in all_feature_cols if _is_float_dtype(dtypes[c])]
    feature_int_cols = [c for c in all_feature_cols if _is_integer_dtype(dtypes[c])]
    bool_cols = [c for c in all_feature_cols if _is_boolean_dtype(dtypes[c])]
    string_cols = [c for c in all_feature_cols if _is_string_dtype(dtypes[c])]
    date_cols = [c for c in all_feature_cols if _is_date_dtype(dtypes[c])]
    categorical_cols = [c for c in all_feature_cols if _is_categorical_dtype(dtypes[c])]
    
    
    # Numerical features: Float + Int + Boolean (as 0/1) + Date (as timestamp) - FEATURES ONLY
    numerical_feature_cols = feature_float_cols + feature_int_cols + bool_cols + date_cols
    
    # Categorical features: String + existing Categorical - FEATURES ONLY
    categorical_feature_cols = string_cols + categorical_cols
    
    # Create feature_dtypes mapping for training features
    all_feature_cols = numerical_feature_cols + categorical_feature_cols
    feature_dtypes = {col: str(dtypes[col]) for col in all_feature_cols}
    
    if verbose:
        print(f"📊 Feature type breakdown:")
        print(f"  Float: {len(feature_float_cols)} columns")
        print(f"  Int: {len(feature_int_cols)} columns") 
        print(f"  Boolean: {len(bool_cols)} columns")
        print(f"  String: {len(string_cols)} columns")
        print(f"  Date: {len(date_cols)} columns")
        print(f"  Categorical: {len(categorical_cols)} columns")
        print(f"  Total features: {len(all_feature_cols)} columns")
        print(f"  Features with recorded dtypes: {len(feature_dtypes)}")

    # Compute scaling parameters using centralized function
    scaling_params = compute_scaling_parameters(
        df.select(numerical_feature_cols), apply_scaling=apply_scaling_parameters, verbose=verbose
    )

    # Compute categorical info and mappings
    cat_feature_info: Dict[str, int] = {}
    category_mappings: Dict[str, Dict[Any, int]] = {}
    for col in categorical_feature_cols:
        if col in schema_map:
            # Use predefined enum from CATEGORICAL_SCHEMAS if available
            if hasattr(mlBridgeLib, 'CATEGORICAL_SCHEMAS') and col in mlBridgeLib.CATEGORICAL_SCHEMAS:
                # Get all possible values from the enum
                enum_def = mlBridgeLib.CATEGORICAL_SCHEMAS[col]
                if hasattr(enum_def, 'categories'):
                    values = list(enum_def.categories)
                    if verbose:
                        print(f"  Using predefined enum for '{col}': {len(values)} categories")
                else:
                    # Fallback to data if enum doesn't have categories
                    col_data = df[col]
                    dt = dtypes[col]
                    if _is_string_dtype(dt):
                        uniques = col_data.fill_null("__NULL__").unique()
                    else:
                        uniques = col_data.drop_nulls().unique()
                    values = uniques.to_list() if hasattr(uniques, 'to_list') else list(uniques)
                    values = sorted(values, key=lambda x: str(x) if x is not None else "")
            else:
                # No predefined enum - infer from data
                col_data = df[col]
                dt = dtypes[col]
                # Handle String columns - include null handling
                if _is_string_dtype(dt):
                    uniques = col_data.fill_null("__NULL__").unique()
                else:
                    # Categorical columns - standard handling
                    uniques = col_data.drop_nulls().unique()
                
                values = uniques.to_list() if hasattr(uniques, 'to_list') else list(uniques)
                # Sort for deterministic ordering
                values = sorted(values, key=lambda x: str(x) if x is not None else "")
                if verbose:
                    print(f"  Inferred categories from data for '{col}': {len(values)} unique values")
            
            mapping = {v: i for i, v in enumerate(values)}
            category_mappings[col] = mapping
            cat_feature_info[col] = len(values)  # unknown will map to this index

    # Infer model type from y dtype
    target_dtype = dtypes[y_name]
    inferred_model_type = 'regression' if _is_float_dtype(target_dtype) else 'classification'

    schema = {
        "schema_version": schema_version,
        "target_column": y_name,
        "feature_dtypes": feature_dtypes,
        "mlp_layers": layers,
        "mlp_dropout": dropout,
        "scaling_params": scaling_params,
        "apply_scaling": apply_scaling_parameters,
        "y_range": list(y_range) if y_range is not None else None,
        "cat_feature_info": cat_feature_info,
        "numerical_feature_cols": numerical_feature_cols,
        "categorical_feature_cols": categorical_feature_cols,
        "model_type": inferred_model_type,
        "category_mappings": category_mappings,
    }
    # Persist basic provenance to the schema for downstream consumers
    schema.update({
        "saved_models_path": str(pathlib.Path(saved_models_path)),
        "model_name": model_name,
    })
    json.dump(schema, open(schema_file, "w"))

    if verbose:
        print(f"✅ Schema saved to {schema_file}")
        print(f"  Numerical features: {len(numerical_feature_cols)}")
        print(f"  Categorical features ({len(cat_feature_info)}): {list(cat_feature_info.keys())}")
        for col, card in cat_feature_info.items():
            print(f"    - {col}: cardinality={card}")

    return schema


def get_model_type_from_schema(
    saved_models_path: pathlib.Path,
    model_name: str,
) -> str:
    """Return model type strictly from saved schema or checkpoint heuristics."""
    sp = resolve_schema_path(saved_models_path, model_name)
    if not sp.exists():
        raise FileNotFoundError(f"Schema file not found: {sp}")
    schema = json.load(open(sp))
    mt = schema.get('model_type')
    if isinstance(mt, str) and mt in ('regression', 'classification'):
        return mt
    raise ValueError("model_type cannot be inferred from schema; set schema['model_type']")


def infer_model_type_from_df(
    df: pl.DataFrame,
    y_name: str,
) -> str:
    """Infer model type from target dtype only (training-time)."""
    if y_name not in df.columns:
        sample_cols = df.columns[:20]
        raise ValueError(
            f"y_name '{y_name}' not found in df columns. "
            f"Columns count={len(df.columns)}; sample={sample_cols}."
        )
    dtype = df[y_name].dtype
    try:
        if dtype.is_float():
            return 'regression'
    except Exception:
        pass
    return 'classification'


def generate_and_save_schema(
    df: pl.DataFrame,
    saved_models_path: pathlib.Path,
    model_name: str,
    y_name: str,
    # 🔧 SHARED PARAMETERS: Must match between training and inference
    layers: Optional[List[int]] = [1024, 512, 256, 128],
    dropout: float = 0.1,
    apply_scaling_parameters: bool = True,
    y_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    # Schema generation parameters
    schema_version: str = "1.1",
    blacklist_patterns: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Generate and save schema with explicit shared parameters.
    
    🔧 SHARED PARAMETERS (must match training/inference):
    - layers: Network architecture [1024, 512, 256, 128]
    - dropout: Dropout rate for model structure
    - apply_scaling_parameters: Whether to apply feature scaling
    - y_range: Output clamping range for regression (None = no clamping)
    
    Schema generation parameters:
    - schema_version: Schema format version
    - blacklist_patterns: Column exclusion patterns
    - verbose: Print generation details
    """
    model_type = infer_model_type_from_df(df=df, y_name=y_name)
    if model_type == 'regression':
        return generate_and_save_schema_regression(
            df=df,
            saved_models_path=saved_models_path,
            model_name=model_name,
            y_name=y_name,
            layers=layers,
            dropout=dropout,
            apply_scaling_parameters=apply_scaling_parameters,
            y_range=y_range,
            schema_version=schema_version,
            blacklist_patterns=blacklist_patterns,
            verbose=verbose,
        )
    else:
        return generate_and_save_schema_classification(
            df=df,
            saved_models_path=saved_models_path,
            model_name=model_name,
            y_name=y_name,
            layers=layers,
            dropout=dropout,
            apply_scaling_parameters=apply_scaling_parameters,
            schema_version=schema_version,
            blacklist_patterns=blacklist_patterns,
            verbose=verbose,
        )

def generate_and_save_schema_regression(
    df: pl.DataFrame,
    saved_models_path: pathlib.Path,
    model_name: str,
    y_name: str,
    layers: Optional[List[int]] = [1024, 512, 256, 128],
    dropout: float = 0.1,
    apply_scaling_parameters: bool = True,
    verbose: bool = False,
    y_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    schema_version: str = "1.1",
    blacklist_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build and persist a schema for regression tasks. Returns the schema dict.
    """
    schema = generate_and_save_schema_core(
        df=df,
        saved_models_path=saved_models_path,
        model_name=model_name,
        y_name=y_name,
        layers=layers,
        dropout=dropout,
        apply_scaling_parameters=apply_scaling_parameters,
        verbose=verbose,
        y_range=y_range,
        schema_version=schema_version,
        blacklist_patterns=blacklist_patterns,
    )
    # Ensure model_type is set
    schema['model_type'] = 'regression'
    json.dump(schema, open(resolve_schema_path(saved_models_path, model_name), 'w'))
    return schema


def generate_and_save_schema_classification(
    df: pl.DataFrame,
    saved_models_path: pathlib.Path,
    model_name: str,
    y_name: str,
    # 🔧 SHARED PARAMETERS: Must match between training and inference
    layers: Optional[List[int]] = [1024, 512, 256, 128],
    dropout: float = 0.1,
    apply_scaling_parameters: bool = True,
    # Schema generation parameters
    schema_version: str = "1.1",
    blacklist_patterns: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Build and persist a schema for classification tasks.
    Adds class vocabulary and class_to_idx mapping to the saved schema.
    Returns the schema dict.
    """
    # First, build the base schema (shares feature/scaling/mappings fields)
    schema = generate_and_save_schema_core(
        df=df,
        saved_models_path=saved_models_path,
        model_name=model_name,
        y_name=y_name,
        layers=layers,
        dropout=dropout,
        apply_scaling_parameters=apply_scaling_parameters,
        verbose=verbose,
        y_range=(0.0, 1.0),  # unused for classification
        schema_version=schema_version,
        blacklist_patterns=blacklist_patterns,
    )

    # Derive classes and mapping
    try:
        labels_series = df[y_name]
        labels_list = labels_series.to_list() if hasattr(labels_series, 'to_list') else list(labels_series)
    except Exception:
        labels_list = list(pl.Series(df[y_name]).to_list())

    # Clean labels: drop None/nulls, ensure deterministic ordering across types by sorting on str
    labels_clean = [c for c in labels_list if c is not None]
    uniq_classes: List[Any] = sorted(set(labels_clean), key=lambda x: str(x))
    if not uniq_classes:
        raise ValueError(f"No valid classes found in column '{y_name}' to build classification schema")
    class_to_idx: Dict[Any, int] = {c: i for i, c in enumerate(uniq_classes)}

    # Persist into schema and re-save
    schema_path = resolve_schema_path(saved_models_path, model_name)
    schema.update({
        "task": "classification",
        "classes": uniq_classes,
        "class_to_idx": class_to_idx,
        "model_type": "classification",
    })
    json.dump(schema, open(schema_path, 'w'))

    if verbose:
        print(f"✅ Classification schema updated with {len(uniq_classes)} classes at {schema_path}")

    return schema

def _train_model_core(
    saved_models_path: pathlib.Path,
    model_name: str,
    train_iterator_fn: Callable,
    validation_iterator_fn: Callable,
    input_dim: int,
    feature_cols: List[str],
    y_name: str,
    layers: List[int] = [1024, 512, 256, 128],
    epochs: int = 3,
    bs: int = 32768,
    device: str = 'cuda',
    y_range: Tuple[float, float] = (0, 1),
    lr: float = 2e-3,
    dropout: float = 0.1,
    weight_decay: float = 1e-4,
    use_amp: bool = True,
    seed: Optional[int] = None,
    verbose: bool = False,
    cat_feature_info: Optional[Dict[str, int]] = None,
    save_model: bool = True,
) -> Tuple[Any, pathlib.Path, Dict[str, Any]]:
    """
    Purpose:
        Shared training core used by both df-based and shard-based trainers.

    Args:
        train_iterator_fn: Function returning training iterator of (xb, yb).
        validation_iterator_fn: Function returning validation iterator (or None).
        input_dim: Number of input features for MLP continuous part.
        saved_models_path: Directory path to save model.
        model_filename: Output model filename.
        feature_cols: Ordered feature names used in training.
        y_name: Target column name.
        layers: MLP hidden layer sizes.
        epochs: Number of training epochs.
        bs: Batch size.
        device: Device string ('cuda' or 'cpu').
        y_range: Clamp range for predictions.
        lr: Learning rate.
        dropout: Dropout rate.
        weight_decay: AdamW weight decay.
        use_amp: Enable automatic mixed precision when available.
        seed: Optional RNG seed.
        verbose: Print detailed logs.
        cat_feature_info: Optional mapping of categorical feature cardinalities.

    Return:
        (model, model_file_path, training_stats)
    """
    if seed is not None:
        torch.manual_seed(seed)

    device_t, scaler, _ = setup_amp(device, use_amp)
    cat_dims = None
    if cat_feature_info:
        # Reserve an extra index for unknowns in embeddings
        cat_dims = [(cardinality + 1, min(50, cardinality // 2)) for cardinality in cat_feature_info.values()]

    model = MLP(input_dim=input_dim, cat_dims=cat_dims, layer_sizes=layers, dropout=dropout).to(device_t)
    # Use SGD for CPU to avoid AdamW CUDA context issues
    if device_t.type == 'cpu':
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🔧 Model created: {total_params:,} parameters, input_dim={input_dim}")
        print(f"🔧 Device: {device_t}, AMP: {use_amp}, Batch size: {bs}")

    best_val = None
    train_samples_total = 0
    val_samples_total = 0

    training_start_time = time.time()
    print(f"▶ Training for {epochs} epoch(s)...")

    for epoch in range(epochs):
        train_iter = train_iterator_fn()
        tloss, tcount = train_one_epoch(model, train_iter, loss_fn, opt, device_t, y_range, scaler, use_amp)
        train_loss = tloss / max(1, tcount)
        train_samples_total = tcount

        vloss = vsamp = None
        val_pred_mean = val_pred_std = None
        val_iter = validation_iterator_fn() if validation_iterator_fn else None
        if val_iter is not None:
            vtot, vcnt, val_pred_mean, val_pred_std = eval_one_epoch_with_stats(model, val_iter, loss_fn, device_t, y_range)
            vloss = vtot / max(1, vcnt)
            vsamp = vcnt
            val_samples_total = vcnt
            best_val = vloss if (best_val is None or vloss < best_val) else best_val

        if verbose or not validation_iterator_fn:
            line = format_epoch_line_with_stats(epoch, epochs, train_loss, vloss, tcount, vsamp or 0, val_pred_mean, val_pred_std)
            print(line)

    training_time = time.time() - training_start_time

    model_file_path = None
    if save_model:
        model_file_path = resolve_model_path(saved_models_path, model_name)
        model_file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_file_path)
        if verbose:
            print(f"✅ Model saved to: {model_file_path}")
            print(f"✅ Training completed in {training_time:.1f}s")

    stats = make_training_stats(
        total_epochs=epochs,
        best_val_loss=best_val,
        training_time=training_time,
        train_samples=train_samples_total,
        val_samples=val_samples_total,
        feature_cols=feature_cols,
        y_name=y_name,
        input_dim=input_dim,
        device=device_t,
        use_amp=use_amp,
    )

    return model, model_file_path, stats


def df_to_scaled_tensors(df: pl.DataFrame, saved_models_path: pathlib.Path, model_name, y_name: str, scaling_mode: str = 'auto'):
    """
    Purpose:
        Convert a Polars DataFrame to tensors using the provided schema for feature order
        and scaling.

    Args:
        df: Source Polars DataFrame.
        schema_path: Path to schema.json containing feature order and scaling parameters.
        y_name: Target column name.
        scaling_mode: 'auto' to use schema.apply_scaling, 'on' to force scale, 'off' to disable.

    Return:
        Tuple (X_cont, X_cat, y, meta) where meta includes feature lists.
    """
    start_t = time.time()
    print("▶ Converting DataFrame to scaled tensors...")
    schema_path = resolve_schema_path(saved_models_path, model_name)
    schema = json.load(open(schema_path))

    numerical_feature_cols = schema['numerical_feature_cols']
    categorical_feature_cols = schema['categorical_feature_cols']
    feature_column_list = numerical_feature_cols + categorical_feature_cols
    scaling_params = schema.get('scaling_params', {})
    preprocessing = schema.get('preprocessing')

    X_num = df.select(numerical_feature_cols).to_numpy().astype(np.float32)
    X_num = np.nan_to_num(X_num, nan=0.0, posinf=1e6, neginf=-1e6)

    # NEW: Apply training-time preprocessing if present in schema
    if preprocessing:
        clip_low, clip_high = preprocessing.get('clip_range', (-100, 100))
        scale_factor = preprocessing.get('scale_factor', 1)
        X_num = np.clip(X_num, clip_low, clip_high)
        if scale_factor and abs(scale_factor) != 1:
            X_num = X_num / float(scale_factor)
    else:
        # Backward compatible mean/std scaling path
        do_scale = (scaling_mode == 'on') or (scaling_mode == 'auto' and schema.get('apply_scaling', False))
        if do_scale and scaling_params:
            X_num = apply_feature_scaling(X_num, numerical_feature_cols, scaling_params, apply_scaling=True)

    X_cont = torch.tensor(X_num, dtype=torch.float32)

    X_cat = None
    if categorical_feature_cols:
        cat_data_list = []
        cat_feature_info = schema.get('cat_feature_info', {})
        category_mappings = schema.get('category_mappings', {})
        # Avoid tqdm/stqdm in Streamlit to prevent StopException noise
        for col in categorical_feature_cols:
            mapping = category_mappings.get(col, {})
            unknown_index = cat_feature_info.get(col, 0)
            mapped_values = (
                df.select(
                    pl.col(col)
                    .replace_strict(mapping, default=unknown_index, return_dtype=pl.Int64)  # vectorized; unseen -> unknown_index
                    .alias(col)
                )
                .to_numpy()
                .ravel()
            )
            cat_data_list.append(mapped_values)
        X_cat = torch.tensor(np.stack(cat_data_list, axis=1), dtype=torch.int64) if cat_data_list else None

    y = torch.tensor(df[y_name].to_numpy(), dtype=torch.float32)
    y = torch.nan_to_num(y, nan=0.5).clamp(0.0, 1.0)

    meta = {
        'feature_column_list': feature_column_list,
        'numerical_feature_cols': numerical_feature_cols,
        'categorical_feature_cols': categorical_feature_cols,
    }
    print(f"✅ Converted to tensors in {time.time()-start_t:.1f}s")
    return X_cont, X_cat, y, meta


def train_regression_from_df(
    df: pl.DataFrame,
    y_names: List[str],
    valid_pct: float = 0.01,
    bs: int = 4096,
    layers: Optional[List[int]] = [2048, 1024, 512, 256],
    epochs: int = 3,
    device: Optional[str] = 'cuda',
    y_range: Optional[Tuple[float, float]] = None,
    lr: float = 1e-3,
    dropout: float = 0.2,
    weight_decay: float = 1e-5,
    use_amp: bool = True,
    verbose: bool = False,
    saved_models_path: Optional[pathlib.Path] = None,
    model_name: Optional[str] = None,
    apply_scaling_parameters: bool = True,
    schema_version: str = "1.1",
) -> Dict[str, Any]:
    torch.manual_seed(42)
    cols = list(df.columns)
    dtypes = {c: str(df.dtypes[i]) for i, c in enumerate(cols)} # todo: use dtype.is_float()
    float_cols = [c for c in cols if 'Float' in dtypes[c]]
    assert len(y_names) == 1, "Only one target column supported"
    y_name = y_names[0]
    assert y_name in float_cols, f"{y_name} must be a float column"
    feature_cols = [c for c in float_cols if c != y_name]
    input_dim = len(feature_cols)

    # Optionally generate and save schema for downstream inference
    if saved_models_path is not None and model_name is not None:
        try:
            _ = generate_and_save_schema(
                df=df,
                saved_models_path=saved_models_path,
                model_name=model_name,
                y_name=y_name,
                layers=layers,
                dropout=dropout,
                apply_scaling_parameters=apply_scaling_parameters,
                verbose=verbose,
                y_range=y_range if y_range is not None else (0.0, 1.0),
                schema_version=schema_version,
            )
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to generate schema: {e}")

    n = df.height
    n_valid = int(max(1, n * valid_pct)) if n > 1 and valid_pct > 0 else 0
    if n_valid > 0:
        train_df = df.slice(0, n - n_valid)
        valid_df = df.slice(n - n_valid, n_valid)
    else:
        train_df, valid_df = df, None

    X_train, y_train = df_to_tensors_regression(train_df, y_name, feature_cols)
    if valid_df is not None:
        X_val, y_val = df_to_tensors_regression(valid_df, y_name, feature_cols)

    device_t = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device_t.type == 'cuda')
        autocast_fn = lambda: torch.amp.autocast('cuda', enabled=use_amp and device_t.type == 'cuda')
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device_t.type == 'cuda')
        autocast_fn = lambda: torch.cuda.amp.autocast(enabled=use_amp and device_t.type == 'cuda')

    model = build_regression_mlp(input_dim, layers, dropout).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=bs, shuffle=True, pin_memory=(device_t.type == 'cuda'))
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=bs, shuffle=False, pin_memory=(device_t.type == 'cuda')) if valid_df is not None else None

    model.train()
    for epoch in range(epochs):
        total_loss, total_count = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device_t)
            yb = yb.to(device_t)
            opt.zero_grad(set_to_none=True)
            with autocast_fn():
                pred = model(xb)
                if y_range is not None:
                    pred = pred.clamp(*y_range)
                mask = torch.isfinite(pred) & torch.isfinite(yb)
                if mask.any():
                    loss = loss_fn(pred[mask], yb[mask])
                else:
                    continue
            if use_amp and device_t.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            total_loss += loss.item() * mask.sum().item()
            total_count += mask.sum().item()
        if verbose:
            avg_loss = total_loss / max(1, total_count)
            # Enhanced logging with validation stats
            val_pred_mean = val_pred_std = None
            if val_loader is not None:
                vtot, vcnt, val_pred_mean, val_pred_std = eval_one_epoch_with_stats(model, val_loader, loss_fn, device_t, y_range)
                vloss = vtot / max(1, vcnt)
                vsamp = vcnt
                line = format_epoch_line_with_stats(
                    epoch, epochs, avg_loss, total_count,
                    vloss, vsamp, val_pred_mean, val_pred_std
                )
            else:
                line = f"Epoch {epoch+1}/{epochs}, train_loss={avg_loss:.6f}, train_samples={total_count}"
            print(line)

    val_loss = None
    if valid_df is not None:
        model.eval()
        with torch.no_grad():
            total_loss, total_count = 0.0, 0
            for xb, yb in val_loader:
                xb = xb.to(device_t)
                yb = yb.to(device_t)
                with autocast_fn():
                    pred = model(xb)
                    if y_range is not None:
                        pred = pred.clamp(*y_range)
                    mask = torch.isfinite(pred) & torch.isfinite(yb)
                    if mask.any():
                        loss = loss_fn(pred[mask], yb[mask])
                        total_loss += loss.item() * mask.sum().item()
                        total_count += mask.sum().item()
            val_loss = total_loss / max(1, total_count)

    return {
        'model': model,
        'feature_cols': feature_cols,
        'y_name': y_name,
        'input_dim': input_dim,
        'val_loss': val_loss,
        'device': str(device_t),
        'use_amp': use_amp,
    }


def train_classification_from_df(
    df: pl.DataFrame,
    y_names: List[str],
    valid_pct: float = 0.01,
    bs: int = 4096,
    layers: Optional[List[int]] = [2048, 1024, 512, 256],
    epochs: int = 3,
    device: Optional[str] = 'cuda',
    lr: float = 1e-3,
    dropout: float = 0.2,
    weight_decay: float = 1e-5,
    use_amp: bool = True,
    verbose: bool = False,
    saved_models_path: Optional[pathlib.Path] = None,
    model_name: Optional[str] = None,
    apply_scaling_parameters: bool = True,
    schema_version: str = "1.1",
) -> Dict[str, Any]:
    torch.manual_seed(42)
    cols = list(df.columns)
    dtypes = {c: str(df.dtypes[i]) for i, c in enumerate(cols)} # todo: use dtype.is_float() and dtype.is_categorical()
    float_cols = [c for c in cols if 'Float' in dtypes[c]]
    assert len(y_names) == 1, "Only one target column supported"
    y_name = y_names[0]
    assert y_name in df.columns, f"{y_name} must be a column in df"

    feature_cols = [c for c in float_cols if c != y_name]
    input_dim = len(feature_cols)
    # Optionally generate and save schema for downstream inference
    if saved_models_path is not None and model_name is not None:
        try:
            _ = generate_and_save_schema(
                df=df,
                saved_models_path=saved_models_path,
                model_name=model_name,
                y_name=y_name,
                layers=layers,
                dropout=dropout,
                apply_scaling_parameters=apply_scaling_parameters,
                verbose=verbose,
                # For classification, y_range is not used; pass default
                y_range=(0.0, 1.0),
                schema_version=schema_version,
            )
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to generate schema: {e}")


    classes = pl.Series(df[y_name]).to_list()
    uniq_classes = sorted(list({c for c in classes}))
    class_to_idx = {c: i for i, c in enumerate(uniq_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    num_classes = len(uniq_classes)

    n = df.height
    n_valid = int(max(1, n * valid_pct)) if n > 1 and valid_pct > 0 else 0
    if n_valid > 0:
        train_df = df.slice(0, n - n_valid)
        valid_df = df.slice(n - n_valid, n_valid)
    else:
        train_df, valid_df = df, None

    X_train, y_train = df_to_tensors_classification(train_df, y_name, feature_cols, class_to_idx)
    if valid_df is not None:
        X_val, y_val = df_to_tensors_classification(valid_df, y_name, feature_cols, class_to_idx)

    device_t = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device_t.type == 'cuda')
        autocast_fn = lambda: torch.amp.autocast('cuda', enabled=use_amp and device_t.type == 'cuda')
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device_t.type == 'cuda')
        autocast_fn = lambda: torch.cuda.amp.autocast(enabled=use_amp and device_t.type == 'cuda')

    model = build_classifier_mlp(input_dim, num_classes, layers, dropout).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=bs, shuffle=True, pin_memory=(device_t.type == 'cuda'))
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=bs, shuffle=False, pin_memory=(device_t.type == 'cuda')) if valid_df is not None else None

    model.train()
    for epoch in range(epochs):
        total_loss, total_count, correct = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device_t)
            yb = yb.to(device_t)
            opt.zero_grad(set_to_none=True)
            with autocast_fn():
                logits = model(xb)
                loss = loss_fn(logits, yb)
            if use_amp and device_t.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            batch_size = yb.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == yb).sum().item()
        if verbose:
            avg_loss = total_loss / max(1, total_count)
            acc = correct / max(1, total_count)
            print(f"Epoch {epoch+1}/{epochs} train_loss={avg_loss:.6f} acc={acc:.4f}")

    val_metrics: Optional[Dict[str, float]] = None
    if valid_df is not None:
        model.eval()
        with torch.no_grad():
            total_loss, total_count, correct = 0.0, 0, 0
            for xb, yb in val_loader:
                xb = xb.to(device_t)
                yb = yb.to(device_t)
                with autocast_fn():
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                batch_size = yb.size(0)
                total_loss += loss.item() * batch_size
                total_count += batch_size
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == yb).sum().item()
            val_metrics = {
                'val_loss': total_loss / max(1, total_count),
                'val_acc': correct / max(1, total_count),
            }

    return {
        'model': model,
        'feature_cols': feature_cols,
        'y_name': y_name,
        'classes': uniq_classes,
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'device': str(device_t),
        'use_amp': use_amp,
        **(val_metrics or {}),
    }


def train_model_regression_from_tensors(
    saved_models_path: pathlib.Path,
    model_name: str,
    Xc_train: torch.Tensor,
    Xk_train: Optional[torch.Tensor],
    y_train: torch.Tensor,
    y_name: str,
    Xc_val: Optional[torch.Tensor] = None,
    Xk_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    bs: int = 32768,
    layers: Optional[List[int]] = [1024, 512, 256, 128],
    epochs: int = 3,
    device: Optional[str] = 'cuda',
    y_range: Tuple[float, float] = (0, 1),
    lr: float = 2e-3,
    dropout: float = 0.1,
    weight_decay: float = 1e-4,
    use_amp: bool = True,
    verbose: bool = True,
    feature_cols: Optional[List[str]] = None,
):
    # input_dim = continuous-only; embeddings handled inside the model
    input_dim = int(Xc_train.shape[1]) if Xc_train.ndim == 2 else 0

    # Load schema to pass categorical info so MLP builds embeddings
    schema_path = resolve_schema_path(saved_models_path, model_name)
    schema_obj = json.load(open(schema_path))
    cat_feature_info = schema_obj.get('cat_feature_info', {})

    # Feature names (for stats only)
    if feature_cols is None:
        feature_cols = [f"f{i}" for i in range(input_dim)]

    # Iterators yield (x_cont, x_cat, y) batches
    def train_iterator_fn():
        return iterate_batches(Xc_train, Xk_train, y_train, bs)

    def validation_iterator_fn():
        if Xc_val is not None and y_val is not None:
            return iterate_batches(Xc_val, Xk_val, y_val, bs)
        return None

    return _train_model_core(
        train_iterator_fn=train_iterator_fn,
        validation_iterator_fn=validation_iterator_fn,
        input_dim=input_dim,
        saved_models_path=pathlib.Path(saved_models_path),
        model_name=model_name,
        feature_cols=feature_cols,
        y_name=y_name,
        layers=layers or [1024, 512, 256, 128],
        epochs=epochs,
        bs=bs,
        device=device or 'cpu',
        y_range=y_range,
        lr=lr,
        dropout=dropout,
        weight_decay=weight_decay,
        use_amp=use_amp,
        seed=None,
        verbose=verbose,
        cat_feature_info=cat_feature_info,  # enable embeddings in the model
    )


def train_classification_from_tensors(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    bs: int = 4096,
    layers: Optional[List[int]] = [2048, 1024, 512, 256],
    epochs: int = 3,
    device: Optional[str] = 'cuda',
    lr: float = 1e-3,
    dropout: float = 0.2,
    weight_decay: float = 1e-5,
    use_amp: bool = True,
    verbose: bool = False,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    assert X_train.dtype == torch.float32, "X_train must be float32"
    assert y_train.dtype in (torch.int64, torch.long), "y_train must be LongTensor of class indices"
    input_dim = X_train.shape[1]
    if num_classes is None:
        num_classes = int(y_train.max().item()) + 1

    device_t = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device_t.type == 'cuda')
        autocast_fn = lambda: torch.amp.autocast('cuda', enabled=use_amp and device_t.type == 'cuda')
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device_t.type == 'cuda')
        autocast_fn = lambda: torch.cuda.amp.autocast(enabled=use_amp and device_t.type == 'cuda')

    model = build_classifier_mlp(input_dim, num_classes, layers, dropout).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=bs, shuffle=True, pin_memory=(device_t.type == 'cuda'))
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=bs, shuffle=False, pin_memory=(device_t.type == 'cuda')) if X_val is not None and y_val is not None else None

    model.train()
    for epoch in range(epochs):
        total_loss, total_count, correct = 0.0, 0, 0
        for xb, yb in train_loader:
            xb = xb.to(device_t)
            yb = yb.to(device_t)
            opt.zero_grad(set_to_none=True)
            with autocast_fn():
                logits = model(xb)
                loss = loss_fn(logits, yb)
            if use_amp and device_t.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            batch_size = yb.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == yb).sum().item()
        if verbose:
            avg_loss = total_loss / max(1, total_count)
            acc = correct / max(1, total_count)
            print(f"Epoch {epoch+1}/{epochs} train_loss={avg_loss:.6f} acc={acc:.4f}")

    val_metrics: Optional[Dict[str, float]] = None
    if val_loader is not None:
        model.eval()
        with torch.no_grad():
            total_loss, total_count, correct = 0.0, 0, 0
            for xb, yb in val_loader:
                xb = xb.to(device_t)
                yb = yb.to(device_t)
                with autocast_fn():
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                batch_size = yb.size(0)
                total_loss += loss.item() * batch_size
                total_count += batch_size
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == yb).sum().item()
            val_metrics = {
                'val_loss': total_loss / max(1, total_count),
                'val_acc': correct / max(1, total_count),
            }

    return {
        'model': model,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'device': str(device_t),
        'use_amp': use_amp,
        **(val_metrics or {}),
    }


def create_torch_shards(
    df: pl.DataFrame,
    schema: Dict[str, Any],
    shard_rows_count: int = 500_000,
    apply_scaling: bool = True,
) -> List[pathlib.Path]:
    """
    Purpose:
        Full-service writer: create tensor shards for PyTorch training using a provided schema dict
        and save them to disk under '{model_name}_shard_{i:09d}.pt' in schema['saved_models_path'].

    Args:
        df: Source Polars DataFrame with features and target.
        schema: Pre-built schema dict (from generate_and_save_schema_*).
        shard_rows_count: Number of rows per shard dictionary.
        apply_scaling: True to apply schema scaling of numeric columns.

    Return:
        List[pathlib.Path]: paths to written shard files.
    """
    y_name = schema.get('target_column')
    if not y_name:
        raise ValueError("schema missing 'target_column'")

    numerical_feature_cols = schema.get("numerical_feature_cols", [])
    categorical_feature_cols = schema.get("categorical_feature_cols", [])
    category_mappings = schema.get("category_mappings", {})
    cat_feature_info = schema.get("cat_feature_info", {})

    # Output location
    saved_models_path = pathlib.Path(schema.get('saved_models_path', '.'))
    model_name = schema.get('model_name')
    if not model_name:
        raise ValueError("schema missing 'model_name'")
    saved_models_path.mkdir(parents=True, exist_ok=True)

    # Remove old processed shards for this model
    old = list(saved_models_path.glob(f"{model_name}_shard_*.pt"))
    if old:
        for f in old:
            try:
                f.unlink()
            except Exception:
                pass

    # Decide scaling behavior
    scaling_params = schema.get("scaling_params", {})
    schema_wants_scaling = bool(schema.get("apply_scaling", True))
    do_scale =  apply_scaling and scaling_params and schema_wants_scaling

    # Task detection (use model_type; fallback to class_to_idx presence)
    is_classification = (schema.get('model_type') == 'classification') or ('class_to_idx' in schema)
    class_to_idx: Dict[Any, int] = schema.get('class_to_idx', {}) if is_classification else {}

    written: List[pathlib.Path] = []
    shard_index = 0
    for start in tqdm(range(0, df.height, shard_rows_count), desc="creating shards"):
        end = min(start + shard_rows_count, df.height)
        shard_df = df.slice(start, end - start)

        tensors: Dict[str, torch.Tensor] = {}

        # Numeric: clean NaNs/Infs and optionally scale
        numerical_data = shard_df.select(numerical_feature_cols).to_numpy().astype(np.float32)
        np.nan_to_num(numerical_data, nan=0.0, posinf=1e6, neginf=-1e6, copy=False)

        if do_scale:
            for j, col in enumerate(numerical_feature_cols):
                p = scaling_params.get(col)
                if not p:
                    continue
                mean = float(p.get("mean", 0.0))
                std = float(p.get("std", 1.0)) or 1.0
                numerical_data[:, j] = (numerical_data[:, j] - mean) / std

        tensors["X_cont"] = torch.from_numpy(numerical_data)

        # Categorical: map to indices
        categorical_data_list: List[np.ndarray] = []
        for col in categorical_feature_cols:
            col_series = shard_df[col]
            mapping = category_mappings.get(col, {})
            unknown_idx = int(cat_feature_info.get(col, len(mapping)))
            mapped_values = (
                col_series
                .replace_strict(mapping, default=unknown_idx, return_dtype=pl.Int64)
                .to_numpy()
                .astype(np.int64)
            )
            categorical_data_list.append(mapped_values)

        if categorical_data_list:
            tensors["X_cat"] = torch.tensor(np.stack(categorical_data_list, axis=1), dtype=torch.int64)
        else:
            tensors["X_cat"] = torch.empty((shard_df.height, 0), dtype=torch.int64)

        # Target
        if is_classification:
            # Map labels to class indices using schema mapping; unseen -> 0 by default
            unknown_class_index = 0
            if class_to_idx:
                try:
                    mapped = (
                        shard_df.select(
                            pl.col(y_name)
                            .replace_strict(class_to_idx, default=unknown_class_index, return_dtype=pl.Int64)
                            .alias(y_name)
                        )
                        .to_numpy()
                        .ravel()
                        .astype(np.int64)
                    )
                except Exception:
                    # Fallback to Python mapping if replace_strict incompatible
                    labels_np = shard_df[y_name].to_numpy()
                    mapped = np.array([class_to_idx.get(v, unknown_class_index) for v in labels_np], dtype=np.int64)
            else:
                # No mapping present; attempt to ordinal-encode deterministically by string
                labels_np = shard_df[y_name].to_numpy()
                uniq = sorted(set([v for v in labels_np if v is not None]), key=lambda x: str(x))
                local_map = {c: i for i, c in enumerate(uniq)}
                mapped = np.array([local_map.get(v, 0) for v in labels_np], dtype=np.int64)
            tensors["y"] = torch.from_numpy(mapped)
        else:
            y = torch.tensor(shard_df[y_name].to_numpy(), dtype=torch.float32)
            y = torch.nan_to_num(y, nan=0.5).clip(0.0, 1.0)
            tensors["y"] = y

        tensors["meta"] = {"scaled": bool(do_scale)}

        shard_path = resolve_shards_path(saved_models_path, model_name, shard_index)
        torch.save(tensors, shard_path)
        written.append(shard_path)
        shard_index += 1

    return written


def iter_torch_shards_raw(
    df: pl.DataFrame,
    cont_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    row_id_col: Optional[str] = None,
    shard_rows_count: int = 500_000,
) -> Iterable[Dict[str, Any]]:
    """
    Create model-agnostic raw shards containing unscaled numeric features and
    per-shard categorical vocabularies.

    Each shard dict contains:
      - 'X_cont_raw': torch.FloatTensor [N, C]
      - 'X_cat_idx': torch.LongTensor [N, K] (indices referencing 'cat_vocab')
      - 'cat_vocab': Dict[col, List[Any]] per shard
      - 'cont_cols', 'cat_cols': List[str]
      - 'row_ids' (optional): torch.LongTensor [N]
      - 'meta': { 'raw': True }
    """
    # Auto-infer columns if not provided: numeric (Float/Int) as continuous, Categorical as categorical
    if not cont_cols or not cat_cols:
        cols = list(df.columns)
        dtypes_map = {c: str(df.dtypes[i]) for i, c in enumerate(cols)} # todo: use dtype.is_float() and dtype.is_categorical()
        inferred_cont = [c for c in cols if ('Float' in dtypes_map[c] or 'Int' in dtypes_map[c]) and c != (row_id_col or '')]
        inferred_cat = [c for c in cols if 'Categorical' in dtypes_map[c]]
        # If user supplied one list, respect it and fill the other
        if not cont_cols:
            cont_cols = inferred_cont
        if not cat_cols:
            cat_cols = inferred_cat

    for start in tqdm(range(0, df.height, shard_rows_count), desc="creating raw shards"):
        end = min(start + shard_rows_count, df.height)
        shard_df = df.slice(start, end - start)

        # Continuous
        X_cont = shard_df.select(cont_cols).to_numpy().astype(np.float32)
        np.nan_to_num(X_cont, nan=0.0, posinf=1e6, neginf=-1e6, copy=False)

        # Categorical: build per-shard vocab and indices
        cat_vocab: Dict[str, List[Any]] = {}
        cat_arrays: List[np.ndarray] = []
        for col in cat_cols:
            values = shard_df[col].to_list()
            uniq = list(dict.fromkeys([v for v in values]))  # preserve order
            vocab_map = {v: i for i, v in enumerate(uniq)}
            cat_vocab[col] = uniq
            idx = np.array([vocab_map.get(v, 0) for v in values], dtype=np.int64)
            cat_arrays.append(idx)
        if cat_arrays:
            X_cat_idx = np.stack(cat_arrays, axis=1)
        else:
            X_cat_idx = np.empty((shard_df.height, 0), dtype=np.int64)

        shard: Dict[str, Any] = {
            'X_cont_raw': torch.from_numpy(X_cont),
            'X_cat_idx': torch.from_numpy(X_cat_idx),
            'cat_vocab': cat_vocab,
            'cont_cols': list(cont_cols),
            'cat_cols': list(cat_cols),
            'meta': {'raw': True},
        }
        if row_id_col and row_id_col in shard_df.columns:
            shard['row_ids'] = torch.tensor(shard_df[row_id_col].to_numpy(), dtype=torch.int64)

        yield shard


def create_torch_shards_raw(
    df: pl.DataFrame,
    saved_models_path: pathlib.Path,
    raw_prefix: str,
    cont_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    row_id_col: Optional[str] = None,
    shard_rows_count: int = 500_000,
) -> pathlib.Path:
    """
    Full-service raw shard creator. Streams shards and writes each to disk immediately.

    Returns list of written shard file paths named '{model_name}_raw_{offset:09d}.pt'.
    """
    saved_models_path.mkdir(parents=True, exist_ok=True)

    # Remove old manifest and any previously written files listed in it
    manifest_path = resolve_raw_path(saved_models_path, raw_prefix)
    if manifest_path.exists():
        try:
            manifest = torch.load(manifest_path, map_location='cpu')
            for p in manifest.get('shards', []):
                try:
                    pathlib.Path(p).unlink()
                except Exception:
                    pass
            manifest_path.unlink()
        except Exception:
            pass

    written: List[pathlib.Path] = []
    shard_index = 0
    for shard in iter_torch_shards_raw(df, cont_cols=cont_cols, cat_cols=cat_cols, row_id_col=row_id_col, shard_rows_count=shard_rows_count):
        shard_path = saved_models_path / f"{raw_prefix}_raw_shard_{shard_index:09d}.pt"
        torch.save(shard, shard_path)
        written.append(shard_path)
        shard_index += 1

    # Save manifest containing shard file paths
    torch.save({'shards': [str(p) for p in written]}, manifest_path)
    print(f"✅ Raw shard creation completed: {len(written)} files, manifest: {manifest_path}")
    return manifest_path


def generate_schema_from_raw_shards(
    raw_manifest_path: pathlib.Path,
    saved_models_path: pathlib.Path,
    model_name: str,
    y_name: str,
    model_type: str,
    layers: Optional[List[int]] = [1024, 512, 256, 128],
    dropout: float = 0.1,
    apply_scaling_parameters: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Build a schema by scanning raw shards listed in a manifest file at raw_manifest_path.
    The manifest contains {'shards': [str(path), ...]} pointing to raw shard .pt files.
    Computes numeric mean/std (if apply_scaling_parameters) and global categorical mappings.
    Persists model_type, target_column, and provenance fields.
    """
    assert model_type in ("regression", "classification")

    manifest = torch.load(raw_manifest_path, map_location='cpu')
    shard_paths: List[pathlib.Path] = [pathlib.Path(p) for p in manifest.get('shards', [])]
    if not shard_paths:
        raise ValueError(f"No shard paths found in manifest: {raw_manifest_path}")

    # Load first shard to discover columns
    first = torch.load(shard_paths[0], map_location='cpu')

    cont_cols = list(first.get('cont_cols', []))
    cat_cols = list(first.get('cat_cols', []))

    # Initialize accumulators
    scaling_params: Dict[str, Dict[str, float]] = {}
    count = 0
    if apply_scaling_parameters and cont_cols:
        x0 = first['X_cont_raw'].numpy().astype(np.float64)
        count = x0.shape[0]
        sums = x0.sum(axis=0)
        sq_sums = (x0 * x0).sum(axis=0)
    else:
        sums = np.zeros(len(cont_cols), dtype=np.float64)
        sq_sums = np.zeros(len(cont_cols), dtype=np.float64)

    category_mappings: Dict[str, Dict[Any, int]] = {}
    cat_feature_info: Dict[str, int] = {}
    cat_vals: Dict[str, List[Any]] = {c: list(first.get('cat_vocab', {}).get(c, [])) for c in cat_cols}
    cat_seen: Dict[str, set] = {c: set(cat_vals[c]) for c in cat_cols}

    # Stream the rest
    for p in shard_paths[1:]:
        shard = torch.load(p, map_location='cpu')
        if apply_scaling_parameters and cont_cols:
            x = shard['X_cont_raw'].numpy().astype(np.float64)
            count += x.shape[0]
            sums += x.sum(axis=0)
            sq_sums += (x * x).sum(axis=0)
        for col in cat_cols:
            vocab = shard.get('cat_vocab', {}).get(col, [])
            for v in vocab:
                if v not in cat_seen[col]:
                    cat_seen[col].add(v)
                    cat_vals[col].append(v)

    if apply_scaling_parameters and cont_cols:
        means = sums / max(1, count)
        vars_ = np.maximum(0.0, (sq_sums / max(1, count)) - (means * means))
        stds = np.sqrt(vars_)
        stds[stds == 0] = 1.0
        for i, col in enumerate(cont_cols):
            scaling_params[col] = {"mean": float(means[i]), "std": float(stds[i])}

    for col in cat_cols:
        vals = cat_vals.get(col, [])
        mapping = {v: i for i, v in enumerate(vals)}
        category_mappings[col] = mapping
        cat_feature_info[col] = len(vals)

    # Create feature_dtypes mapping - assume all cont_cols are Float32 and cat_cols are Categorical
    all_feature_cols = cont_cols + cat_cols
    feature_dtypes = {}
    for col in cont_cols:
        feature_dtypes[col] = "Float32"
    for col in cat_cols:
        feature_dtypes[col] = "Categorical"

    schema = {
        "schema_version": "1.1",
        "target_column": y_name,
        "feature_dtypes": feature_dtypes,
        "mlp_layers": layers,
        "mlp_dropout": dropout,
        "scaling_params": scaling_params,
        "apply_scaling": bool(apply_scaling_parameters),
        "y_range": [0.0, 1.0],
        "cat_feature_info": cat_feature_info,
        "numerical_feature_cols": cont_cols,
        "categorical_feature_cols": cat_cols,
        "category_mappings": category_mappings,
        "model_type": model_type,
        "saved_models_path": str(pathlib.Path(saved_models_path)),
        "model_name": model_name,
        "raw_manifest_path": str(raw_manifest_path),
    }

    if model_type == 'classification':
        # Classes are not derivable from raw shards; caller should provide or build from labels df later.
        schema.setdefault('classes', [])
        schema.setdefault('class_to_idx', {})

    # Persist
    saved_models_path.mkdir(parents=True, exist_ok=True)
    json.dump(schema, open(resolve_schema_path(saved_models_path, model_name), 'w'))
    if verbose:
        print(f"✅ Schema saved to {resolve_schema_path(saved_models_path, model_name)}")

    return schema


def shard_batch_iterator_with_schema(
    raw_shard_files: List[pathlib.Path],
    schema: Dict[str, Any],
    bs: int,
    y_provider: Optional[Callable[[np.ndarray], np.ndarray]] = None,
):
    """
    Iterator over raw shards applying schema scaling/mappings and fetching y on-the-fly.
    - y_provider: function(row_ids_np) -> y_np, required if shards contain 'row_ids' and y is not present.
    """
    cont_cols = schema.get('numerical_feature_cols', [])
    cat_cols = schema.get('categorical_feature_cols', [])
    scaling_params = schema.get('scaling_params', {})
    apply_scaling = bool(schema.get('apply_scaling', False))
    category_mappings = schema.get('category_mappings', {})
    class_to_idx = schema.get('class_to_idx', {})
    is_classification = (schema.get('model_type') == 'classification') or bool(class_to_idx)
    y_name = schema.get('target_column')

    means = np.array([scaling_params.get(c, {}).get('mean', 0.0) for c in cont_cols], dtype=np.float32)
    stds = np.array([scaling_params.get(c, {}).get('std', 1.0) or 1.0 for c in cont_cols], dtype=np.float32)

    for f in raw_shard_files:
        payload = torch.load(f, map_location='cpu')
        X_cont = payload['X_cont_raw'].numpy().astype(np.float32)
        X_cat_idx = payload['X_cat_idx'].numpy().astype(np.int64)
        cat_vocab = payload.get('cat_vocab', {})
        row_ids = payload.get('row_ids', None)

        # Numeric scaling
        if apply_scaling and cont_cols:
            X_cont = (X_cont - means) / stds

        # Remap per-shard cat indices to global indices
        if X_cat_idx.shape[1] == len(cat_cols) and len(cat_cols) > 0:
            remapped_cols = []
            for j, col in enumerate(cat_cols):
                shard_vocab = cat_vocab.get(col, [])
                global_map = category_mappings.get(col, {})
                # Build a fast map shard_idx -> global_idx
                translate = np.array([global_map.get(v, 0) for v in shard_vocab], dtype=np.int64)
                remapped = translate[X_cat_idx[:, j]] if translate.size > 0 else np.zeros(X_cat_idx.shape[0], dtype=np.int64)
                remapped_cols.append(remapped)
            X_cat_global = np.stack(remapped_cols, axis=1)
        else:
            X_cat_global = np.empty((X_cont.shape[0], 0), dtype=np.int64)

        # Fetch y
        if 'y' in payload:
            y_np = payload['y'].numpy()
        else:
            assert y_provider is not None, "y_provider is required when raw shards do not include 'y'"
            assert row_ids is not None, "row_ids required in raw shards to fetch y"
            row_ids_np = row_ids.numpy()
            y_np = y_provider(row_ids_np)
        if is_classification:
            # ensure indices
            if y_np.dtype != np.int64:
                # If labels provided, map via class_to_idx; else assume already indices
                if class_to_idx:
                    y_np = np.array([class_to_idx.get(v, 0) for v in y_np], dtype=np.int64)
                else:
                    y_np = y_np.astype(np.int64)
        else:
            y_np = y_np.astype(np.float32)

        # Yield in batches
        for i in range(0, X_cont.shape[0], bs):
            xb_cont = torch.tensor(X_cont[i:i+bs], dtype=torch.float32)
            xb_cat = torch.tensor(X_cat_global[i:i+bs], dtype=torch.int64) if X_cat_global.size else None
            yb = torch.tensor(y_np[i:i+bs], dtype=(torch.long if is_classification else torch.float32))
            yield xb_cont, xb_cat, yb


def train_regression_from_raw_shards(
    schema: Dict[str, Any],
    valid_pct: float = 0.01,
    bs: int = 32768,
    layers: Optional[List[int]] = [1024, 512, 256, 128],
    epochs: int = 3,
    device: Optional[str] = 'cuda',
    y_range: Tuple[float, float] = (0, 1),
    lr: float = 2e-3,
    seed: int = 42,
    use_amp: bool = True,
    dropout: float = 0.1,
    weight_decay: float = 1e-4,
    verbose: bool = False,
) -> Tuple[Any, Optional[pathlib.Path], Dict[str, Any]]:
    # Discover raw shard files from schema provenance
    saved_models_path = pathlib.Path(schema.get('saved_models_path', '.'))
    model_name = schema.get('model_name')
    if not model_name:
        raise ValueError("schema missing 'model_name'")
    # Prefer explicit raw_manifest_path in schema; else fall back to manifest naming convention
    raw_manifest = schema.get('raw_manifest_path')
    if raw_manifest and pathlib.Path(raw_manifest).exists():
        manifest_path = pathlib.Path(raw_manifest)
        raw_shard_files = [pathlib.Path(p) for p in torch.load(manifest_path, map_location='cpu').get('shards', [])]
    else:
        raw_shard_files = resolve_raw_shard_files(saved_models_path, model_name)

    # Infer input dim from first shard
    payload0 = torch.load(raw_shard_files[0], map_location='cpu')
    Xc0 = payload0['X_cont_raw']
    input_dim = int(Xc0.shape[1])

    n_valid = 1 if valid_pct > 0 and len(raw_shard_files) > 1 else 0
    train_files = raw_shard_files[:-n_valid] if n_valid else raw_shard_files
    valid_files = raw_shard_files[-n_valid:] if n_valid else []

    def train_iterator_fn():
        return shard_batch_iterator_with_schema(train_files, schema, bs)

    def validation_iterator_fn():
        if valid_files:
            return shard_batch_iterator_with_schema(valid_files, schema, bs)
        return None

    feature_cols = schema.get('numerical_feature_cols', []) + schema.get('categorical_feature_cols', [])

    # Model output path from schema provenance
    saved_models_path = pathlib.Path(schema.get('saved_models_path', '.'))
    model_name = schema.get('model_name')

    return _train_model_core(
        train_iterator_fn=train_iterator_fn,
        validation_iterator_fn=validation_iterator_fn,
        input_dim=input_dim,
        saved_models_path=saved_models_path,
        model_name=model_name,
        feature_cols=feature_cols,
        y_name=schema.get('target_column', 'Target'),
        layers=layers,
        epochs=epochs,
        bs=bs,
        device=device,
        y_range=y_range,
        lr=lr,
        dropout=dropout,
        weight_decay=weight_decay,
        use_amp=use_amp,
        seed=seed,
        verbose=verbose,
        cat_feature_info=schema.get('cat_feature_info', {}),
    )


def train_classification_from_raw_shards(
    schema: Dict[str, Any],
    valid_pct: float = 0.01,
    bs: int = 4096,
    layers: Optional[List[int]] = None,
    epochs: int = 3,
    device: Optional[str] = 'cuda',
    lr: float = 1e-3,
    dropout: float = 0.2,
    weight_decay: float = 1e-5,
    use_amp: bool = True,
    verbose: bool = False,
) -> Tuple[Any, Optional[pathlib.Path], Dict[str, Any]]:
    # Discover raw shard files from schema provenance
    saved_models_path = pathlib.Path(schema.get('saved_models_path', '.'))
    model_name = schema.get('model_name')
    if not model_name:
        raise ValueError("schema missing 'model_name'")
    raw_manifest = schema.get('raw_manifest_path')
    if raw_manifest and pathlib.Path(raw_manifest).exists():
        manifest_path = pathlib.Path(raw_manifest)
        raw_shard_files = [pathlib.Path(p) for p in torch.load(manifest_path, map_location='cpu').get('shards', [])]
    else:
        raw_shard_files = resolve_raw_shard_files(saved_models_path, model_name)

    payload0 = torch.load(raw_shard_files[0], map_location='cpu')
    Xc0 = payload0['X_cont_raw']
    Xk0 = payload0.get('X_cat_idx', None)
    input_dim = int(Xc0.shape[1]) + (int(Xk0.shape[1]) if (Xk0 is not None and Xk0.numel() > 0) else 0)

    n_valid = 1 if valid_pct > 0 and len(raw_shard_files) > 1 else 0
    train_files = raw_shard_files[:-n_valid] if n_valid else raw_shard_files
    valid_files = raw_shard_files[-n_valid:] if n_valid else []

    def _iter(files: List[pathlib.Path]):
        for xb_cont, xb_cat, yb in shard_batch_iterator_with_schema(files, schema, bs):
            # For classifier without embeddings we pass concatenated inputs
            if xb_cat is not None and xb_cat.numel() > 0:
                yield torch.cat([xb_cont, xb_cat.float()], dim=1), yb
            else:
                yield xb_cont, yb

    train_iterator_fn = lambda: _iter(train_files)
    val_iterator_fn = (lambda: _iter(valid_files)) if valid_files else None

    stats = train_classification_from_iterator(
        train_iterator_fn=train_iterator_fn,
        val_iterator_fn=val_iterator_fn,
        input_dim=input_dim,
        num_classes=max(1, len(schema.get('classes', [])) or  int(max(1, 1))),
        epochs=epochs,
        device=device,
        lr=lr,
        layers=layers,
        dropout=dropout,
        weight_decay=weight_decay,
        use_amp=use_amp,
        verbose=verbose,
    )

    model = stats['model']
    model_file_path = resolve_model_path(saved_models_path, model_name)
    model_file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_file_path)

    training_stats = make_training_stats(
        total_epochs=epochs,
        best_val_loss=stats.get('val_loss'),
        training_time=None,
        train_samples=0,
        val_samples=0,
        feature_cols=schema.get('numerical_feature_cols', []) + schema.get('categorical_feature_cols', []),
        y_name=schema.get('target_column', 'Target'),
        input_dim=input_dim,
        device=torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu'),
        use_amp=stats.get('use_amp', True),
    )

    return model, model_file_path, training_stats
def train_model_regression_from_shards(
    schema: Dict[str, Any],
    # 🚫 TRAINING-ONLY PARAMETERS: Do not affect inference
    valid_pct: float = 0.01,
    bs: int = 32768,
    epochs: int = 3,
    device: str = 'cuda',
    lr: float = 2e-3,
    seed: int = 42,
    use_amp: bool = True,
    weight_decay: float = 1e-4,
    verbose: bool = False,
    save_model: bool = True,
) -> Tuple[Any, Optional[pathlib.Path], Dict[str, Any]]:
    # Validate schema and target
    y_name = schema.get('target_column')
    if not y_name:
        raise ValueError("schema missing 'target_column'")

    # Verify this is a regression task - target should be float type
    feature_dtypes = schema.get('feature_dtypes', {})
    # For regression, we expect the model_type to be 'regression' 
    model_type = schema.get('model_type', 'regression')
    assert model_type == 'regression', f"Expected regression model, got {model_type}"
    
    # 🔧 GET SHARED PARAMETERS FROM SCHEMA
    # These parameters were set during schema generation and must match inference
    layers = schema.get('mlp_layers', [1024, 512, 256, 128])
    dropout = schema.get('mlp_dropout', 0.1)
    y_range_list = schema.get('y_range', [0, 1])
    y_range = tuple(y_range_list) if y_range_list else None
    apply_scaling = schema.get('apply_scaling', False)
    
    saved_models_path = pathlib.Path(schema.get('saved_models_path', '.'))
    model_name = schema.get('model_name')
    
    if verbose:
        print(f"🔧 Using shared parameters from schema:")
        print(f"   - mlp_layers: {layers}")
        print(f"   - mlp_dropout: {dropout}")
        print(f"   - y_range: {y_range}")
        print(f"   - apply_scaling: {apply_scaling}")

    scaling_params = schema.get('scaling_params', {})
    numerical_feature_cols = schema.get('numerical_feature_cols', [])
    cat_feature_info = schema.get('cat_feature_info', {})

    saved_models_path = pathlib.Path(schema.get('saved_models_path', '.'))
    model_name = schema.get('model_name')
    if not model_name:
        raise ValueError("schema missing 'model_name'")
    shard_files = sorted(saved_models_path.glob(model_name + '_shard_*.pt'))
    assert shard_files, f"No shard_*.pt files in {saved_models_path}"
    sample_payload = torch.load(shard_files[0], map_location='cpu')
    Xc_sample: torch.Tensor = sample_payload.get('X_cont', torch.empty(0))
    input_dim = int(Xc_sample.shape[1]) if Xc_sample.ndim == 2 else 0

    n_valid = 1 if valid_pct > 0 and len(shard_files) > 1 else 0
    train_files = shard_files[:-n_valid] if n_valid else shard_files
    valid_files = shard_files[-n_valid:] if n_valid else []

    def train_iterator_fn():
        return shard_batch_iterator(train_files, bs)

    def validation_iterator_fn():
        if valid_files:
            return shard_batch_iterator(valid_files, bs)
        return None

    # Get feature columns from schema
    numerical_feature_cols = schema.get('numerical_feature_cols', [])
    categorical_feature_cols = schema.get('categorical_feature_cols', [])
    code_mappings = schema.get('code_mappings', [])
    feature_cols = numerical_feature_cols + categorical_feature_cols

    return _train_model_core(
        train_iterator_fn=train_iterator_fn,
        validation_iterator_fn=validation_iterator_fn,
        input_dim=input_dim,
        saved_models_path=saved_models_path,
        model_name=model_name,
        feature_cols=feature_cols,
        y_name=y_name,
        layers=layers,
        epochs=epochs,
        bs=bs,
        device=device,
        y_range=y_range,
        lr=lr,
        dropout=dropout,
        weight_decay=weight_decay,
        use_amp=use_amp,
        seed=seed,
        verbose=verbose,
        cat_feature_info=cat_feature_info,
        save_model=save_model,
    )


def train_model_from_shards(
    schema: Dict[str, Any],
    # 🚫 TRAINING-ONLY PARAMETERS: Do not affect inference
    valid_pct: float = 0.01,
    bs: int = 32768,  # Will be adjusted for classification
    epochs: int = 3,
    device: str = 'cuda',
    lr: float = 2e-3,  # Will be adjusted for classification
    seed: Optional[int] = 42,
    use_amp: bool = True,
    weight_decay: float = 1e-4,  # Will be adjusted for classification
    verbose: bool = False,
    save_model: bool = True,
):
    """
    Train model from shards using schema parameters.
    
    🔧 SHARED PARAMETERS come from schema (set during schema generation):
    - layers, dropout, y_range, apply_scaling, features, scaling, mappings
    
    🚫 TRAINING-ONLY PARAMETERS (do not affect inference):
    - valid_pct: Validation split percentage
    - bs: Batch size (auto-adjusted: 32768 for regression, 4096 for classification)
    - epochs: Number of training epochs
    - device: Training device ('cuda' or 'cpu')
    - lr: Learning rate (auto-adjusted: 2e-3 for regression, 1e-3 for classification)
    - seed: Random seed for reproducibility
    - use_amp: Use automatic mixed precision
    - weight_decay: L2 regularization (auto-adjusted: 1e-4 for regression, 1e-5 for classification)
    - verbose: Print training progress
    - save_model: Save trained model to disk
    """
    model_type = schema.get('model_type') or ('classification' if 'class_to_idx' in schema else 'regression')
    if model_type == 'regression':
        return train_model_regression_from_shards(
            schema=schema,
            valid_pct=valid_pct,
            bs=bs,
            epochs=epochs,
            device=device,
            lr=lr,
            seed=seed,
            use_amp=use_amp,
            weight_decay=weight_decay,
            verbose=verbose,
            save_model=save_model,
        )
    else:
        # Auto-adjust parameters for classification
        classification_bs = 4096 if bs == 32768 else bs  # Use default or user override
        classification_lr = 1e-3 if lr == 2e-3 else lr  # Use default or user override
        classification_weight_decay = 1e-5 if weight_decay == 1e-4 else weight_decay  # Use default or user override
        
        return train_model_classification_from_shards(
            schema=schema,
            valid_pct=valid_pct,
            bs=classification_bs,
            epochs=epochs,
            device=device,
            lr=classification_lr,
            seed=seed,
            use_amp=use_amp,
            weight_decay=classification_weight_decay,
            verbose=verbose,
            save_model=save_model,
        )


def train_classification_from_iterator(
    train_iterator_fn: Callable[[], Any],
    val_iterator_fn: Optional[Callable[[], Any]],
    input_dim: int,
    num_classes: int,
    epochs: int = 3,
    device: Optional[str] = 'cuda',
    lr: float = 1e-3,
    layers: Optional[List[int]] = None,
    dropout: float = 0.2,
    weight_decay: float = 1e-5,
    use_amp: bool = True,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Train a classification model using iterator functions for data loading.
    
    Args:
        seed: Optional random seed for reproducible training results.
    """
    device_t = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    
    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device_t.type == 'cuda')
        autocast_fn = lambda: torch.amp.autocast('cuda', enabled=use_amp and device_t.type == 'cuda')
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device_t.type == 'cuda')
        autocast_fn = lambda: torch.cuda.amp.autocast(enabled=use_amp and device_t.type == 'cuda')

    model_layers = layers or [2048, 1024, 512, 256]
    model = build_classifier_mlp(input_dim, num_classes, model_layers, dropout).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss, total_count, correct = 0.0, 0, 0
        for xb, yb in train_iterator_fn():
            xb = xb.to(device_t)
            yb = yb.to(device_t)
            opt.zero_grad(set_to_none=True)
            with autocast_fn():
                logits = model(xb)
                loss = loss_fn(logits, yb)
            if use_amp and device_t.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            batch = yb.size(0)
            total_loss += loss.item() * batch
            total_count += batch
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == yb).sum().item()
        if verbose:
            train_loss = total_loss/max(1,total_count)
            train_acc = correct/max(1,total_count)
            
            # Enhanced logging for classification
            line = f"Epoch {epoch+1}/{epochs}, train_loss={train_loss:.6f}, train_acc={train_acc:.4f}, train_samples={total_count}"
            
            # Add validation stats if available (per-epoch validation)
            if val_iterator_fn is not None:
                model.eval()
                with torch.no_grad():
                    val_loss, val_count, val_acc, val_conf_mean, val_conf_std = eval_classification_with_stats(
                        model, val_iterator_fn(), loss_fn, device_t
                    )
                    val_loss = val_loss / max(1, val_count)
                    line += f", val_loss={val_loss:.6f}, val_acc={val_acc:.4f}, val_samples={val_count}"
                    if val_conf_mean is not None and val_conf_std is not None:
                        line += f", val_conf_mean={val_conf_mean:.4f}, val_conf_std={val_conf_std:.4f}"
                model.train()  # Switch back to training mode
            
            print(line)

    val_metrics: Optional[Dict[str, float]] = None
    if val_iterator_fn is not None:
        model.eval()
        with torch.no_grad():
            total_loss, total_count, correct = 0.0, 0, 0
            for xb, yb in val_iterator_fn():
                xb = xb.to(device_t)
                yb = yb.to(device_t)
                with autocast_fn():
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                batch = yb.size(0)
                total_loss += loss.item() * batch
                total_count += batch
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == yb).sum().item()
            val_metrics = {
                'val_loss': total_loss / max(1, total_count),
                'val_acc': correct / max(1, total_count),
            }

    return {
        'model': model,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'device': str(device_t),
        'use_amp': use_amp,
        **(val_metrics or {}),
    }


def train_model_classification_from_shards(
    schema: Dict[str, Any],
    # 🚫 TRAINING-ONLY PARAMETERS: Do not affect inference
    valid_pct: float = 0.01,
    bs: int = 4096,
    epochs: int = 3,
    device: str = 'cuda',
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    use_amp: bool = True,
    seed: Optional[int] = None,
    verbose: bool = False,
    save_model: bool = True,
) -> Tuple[Any, Optional[pathlib.Path], Dict[str, Any]]:
    """
    Train a classifier from shard files created by create_torch_shards.
    Expects shard tensors with keys X_cont, X_cat (optional), y (Long indices).
    Looks up shards in schema['saved_models_path'] matching f"{schema['model_name']}_shard_*.pt".
    
    Args:
        seed: Optional random seed for reproducible training results.
    """
    saved_models_path = pathlib.Path(schema.get('saved_models_path', '.'))
    model_name = schema.get('model_name')
    if not model_name:
        raise ValueError("schema missing 'model_name'")
    
    # 🔧 GET SHARED PARAMETERS FROM SCHEMA
    # These parameters were set during schema generation and must match inference
    layers = schema.get('mlp_layers', [1024, 512, 256, 128])
    dropout = schema.get('mlp_dropout', 0.1)
    apply_scaling = schema.get('apply_scaling', False)
    
    if verbose:
        print(f"🔧 Using shared parameters from schema:")
        print(f"   - mlp_layers: {layers}")
        print(f"   - mlp_dropout: {dropout}")
        print(f"   - apply_scaling: {apply_scaling}")

    y_name = schema.get('target_column')
    if not y_name:
        raise ValueError("schema missing 'target_column'")
    
    # Verify this is a classification task
    model_type = schema.get('model_type', 'classification')
    assert model_type == 'classification', f"Expected classification model, got {model_type}"

    # Get feature columns from schema
    numerical_feature_cols = schema.get('numerical_feature_cols', [])
    categorical_feature_cols = schema.get('categorical_feature_cols', [])

    # Gather shards
    shard_files = sorted(saved_models_path.glob(model_name + '_shard_*.pt'))
    assert shard_files, f"No shard_*.pt files in {saved_models_path}"
    sample_payload = torch.load(shard_files[0], map_location='cpu')
    Xc_sample: torch.Tensor = sample_payload.get('X_cont', torch.empty(0))
    Xk_sample: Optional[torch.Tensor] = sample_payload.get('X_cat', None)
    y_sample: torch.Tensor = sample_payload.get('y', torch.empty(0, dtype=torch.long))

    input_dim_cont = int(Xc_sample.shape[1]) if Xc_sample.ndim == 2 else 0
    input_dim_cat = int(Xk_sample.shape[1]) if (Xk_sample is not None and Xk_sample.ndim == 2 and Xk_sample.numel() > 0) else 0
    input_dim = input_dim_cont + input_dim_cat
    if y_sample.numel() == 0:
        raise RuntimeError("Sample shard missing 'y' for classification")
    num_classes = int(y_sample.max().item()) + 1

    # Train/val split by shards
    n_valid = 1 if valid_pct > 0 and len(shard_files) > 1 else 0
    train_files = shard_files[:-n_valid] if n_valid else shard_files
    valid_files = shard_files[-n_valid:] if n_valid else []

    def _iter(files: List[pathlib.Path]):
        for f in files:
            X_cont, X_cat, y = load_shard_data(f)
            # For classification, y should be Long
            if y.dtype != torch.long:
                y = y.to(torch.long)
            for i in range(0, X_cont.shape[0], bs):
                xb = X_cont[i:i+bs]
                yb = y[i:i+bs]
                # Build combined input for classifier without embeddings: concat X_cat if present
                if X_cat is not None and X_cat.numel() > 0:
                    xb_full = torch.cat([xb, X_cat[i:i+bs].float()], dim=1)
                else:
                    xb_full = xb
                yield xb_full, yb

    train_iterator_fn = lambda: _iter(train_files)
    val_iterator_fn = (lambda: _iter(valid_files)) if valid_files else None

    stats = train_classification_from_iterator(
        train_iterator_fn=train_iterator_fn,
        val_iterator_fn=val_iterator_fn,
        input_dim=input_dim,
        num_classes=num_classes,
        epochs=epochs,
        device=device,
        lr=lr,
        layers=layers,
        dropout=dropout,
        weight_decay=weight_decay,
        use_amp=use_amp,
        seed=seed,
        verbose=verbose,
    )

    # Save model if requested
    model = stats['model']
    model_file_path: Optional[pathlib.Path] = None
    if save_model:
        model_file_path = resolve_model_path(saved_models_path, model_name)
        model_file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_file_path)

    # Training stats summary similar to regression
    training_stats = make_training_stats(
        total_epochs=epochs,
        best_val_loss=stats.get('val_loss'),
        training_time=None,
        train_samples=0,
        val_samples=0,
        feature_cols=numerical_feature_cols + categorical_feature_cols,
        y_name=y_name,
        input_dim=input_dim,
        device=torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu'),
        use_amp=stats.get('use_amp', True),
    )

    return model, model_file_path, training_stats


def train_model_from_df(
    df: pl.DataFrame,
    y_names: List[str],
    **kwargs,
):
    """Wrapper: choose regression/classification training from a DataFrame by y dtype.

    Assumes a single target in y_names.
    """
    assert len(y_names) == 1, "Only one target column supported"
    y_name = y_names[0]
    model_type = infer_model_type_from_df(df=df, y_name=y_name)
    if model_type == 'regression':
        return train_regression_from_df(
            df=df,
            y_names=y_names,
            valid_pct=kwargs.get('valid_pct', 0.01),
            bs=kwargs.get('bs', 4096),
            layers=kwargs.get('layers', [2048, 1024, 512, 256]),
            epochs=kwargs.get('epochs', 3),
            device=kwargs.get('device', 'cuda'),
            y_range=kwargs.get('y_range'),
            lr=kwargs.get('lr', 1e-3),
            dropout=kwargs.get('dropout', 0.2),
            weight_decay=kwargs.get('weight_decay', 1e-5),
            use_amp=kwargs.get('use_amp', True),
            verbose=kwargs.get('verbose', False),
            saved_models_path=kwargs.get('saved_models_path'),
            model_name=kwargs.get('model_name'),
            apply_scaling_parameters=kwargs.get('apply_scaling_parameters', True),
            schema_version=kwargs.get('schema_version', '1.1'),
        )
    else:
        return train_classification_from_df(
            df=df,
            y_names=y_names,
            valid_pct=kwargs.get('valid_pct', 0.01),
            bs=kwargs.get('bs', 4096),
            layers=kwargs.get('layers', [2048, 1024, 512, 256]),
            epochs=kwargs.get('epochs', 3),
            device=kwargs.get('device', 'cuda'),
            lr=kwargs.get('lr', 1e-3),
            dropout=kwargs.get('dropout', 0.2),
            weight_decay=kwargs.get('weight_decay', 1e-5),
            use_amp=kwargs.get('use_amp', True),
            verbose=kwargs.get('verbose', False),
            saved_models_path=kwargs.get('saved_models_path'),
            model_name=kwargs.get('model_name'),
            apply_scaling_parameters=kwargs.get('apply_scaling_parameters', True),
            schema_version=kwargs.get('schema_version', '1.1'),
        )


def train_model_from_tensors(
    *,
    # Regression tensors
    Xc_train: Optional[torch.Tensor] = None,
    Xk_train: Optional[torch.Tensor] = None,
    y_train: Optional[torch.Tensor] = None,
    Xc_val: Optional[torch.Tensor] = None,
    Xk_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    # Classification tensors
    X_train: Optional[torch.Tensor] = None,
    y_train_cls: Optional[torch.Tensor] = None,
    X_val_cls: Optional[torch.Tensor] = None,
    y_val_cls: Optional[torch.Tensor] = None,
    y_name: Optional[str] = None,
    df_for_infer: Optional[pl.DataFrame] = None,
    # Common kwargs
    **kwargs,
):
    """Wrapper: choose regression/classification training from tensors.

    Inference uses df/y_name when provided; else inspects y tensors' dtype.
    """
    model_type: Optional[str] = None
    if df_for_infer is not None and y_name is not None:
        model_type = infer_model_type_from_df(df=df_for_infer, y_name=y_name)
    elif y_train is not None and (y_train.dtype.is_float()):
        model_type = 'regression'
    elif y_train_cls is not None and (y_train_cls.dtype.is_integer()):
        model_type = 'classification'
    else:
        raise ValueError("Cannot infer model type; provide df_for_infer+y_name or y tensors")

    if model_type == 'regression':
        return train_model_regression_from_tensors(
            saved_models_path=kwargs.get('saved_models_path'),
            model_name=kwargs.get('model_name'),
            Xc_train=Xc_train,
            Xk_train=Xk_train,
            y_train=y_train,
            y_name=y_name or kwargs.get('y_name', 'Target'),
            Xc_val=Xc_val,
            Xk_val=Xk_val,
            y_val=y_val,
            bs=kwargs.get('bs', 32768),
            layers=kwargs.get('layers', [1024, 512, 256, 128]),
            epochs=kwargs.get('epochs', 3),
            device=kwargs.get('device', 'cuda'),
            y_range=kwargs.get('y_range', (0, 1)),
            lr=kwargs.get('lr', 2e-3),
            dropout=kwargs.get('dropout', 0.1),
            weight_decay=kwargs.get('weight_decay', 1e-4),
            use_amp=kwargs.get('use_amp', True),
            verbose=kwargs.get('verbose', True),
            feature_cols=kwargs.get('feature_cols'),
        )
    else:
        return train_classification_from_tensors(
            X_train=X_train,
            y_train=y_train_cls,
            X_val=X_val_cls,
            y_val=y_val_cls,
            bs=kwargs.get('bs', 4096),
            layers=kwargs.get('layers', [2048, 1024, 512, 256]),
            epochs=kwargs.get('epochs', 3),
            device=kwargs.get('device', 'cuda'),
            lr=kwargs.get('lr', 1e-3),
            dropout=kwargs.get('dropout', 0.2),
            weight_decay=kwargs.get('weight_decay', 1e-5),
            use_amp=kwargs.get('use_amp', True),
            verbose=kwargs.get('verbose', False),
            num_classes=kwargs.get('num_classes'),
        )


def predict_regression_model(saved_models_path: pathlib.Path, model_name: str, df: pl.DataFrame, device: str = 'cpu', max_samples: Optional[int] = None, strict_validation: bool = True, parity_drop_non_finite: bool = False) -> pl.DataFrame:
    """
    Embeddings-aware prediction:
    - Builds (X_cont, X_cat) via df_to_scaled_tensors (respects schema scaling).
    - Computes expected_input_dim = num_continuous + sum(embedding_dims).
    - Reads first linear layer in_features (ignoring embedding matrices) for a correct shape check.
    - Runs batched inference passing both x_cont and x_cat to the model.
    """

    # Load model
    model_file = resolve_model_path(saved_models_path, model_name)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    # Determine device and load model accordingly
    device_t = torch.device(device if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_file, map_location=device_t, weights_only=False)

    # Load schema
    schema_path = resolve_schema_path(saved_models_path, model_name)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    schema = json.load(open(schema_path))

    # Validate inference DataFrame dtypes against schema (always strict for Pct_NS)
    y_name = schema.get('target_column') or 'Target'
    strict_local = bool(strict_validation or (str(y_name).lower() == 'pct_ns'))
    df = validate_inference_dataframe_dtypes(df, schema, strict=strict_local, verbose=True)

    # y_name comes from schema (used for naming and optional error calc)
    # already initialized above
    
    # Load y_range from schema for proper clamping
    y_range_list = schema.get('y_range', [0, 1])
    y_range = tuple(y_range_list) if y_range_list else None

    # Check for required feature columns
    numerical_feature_cols = schema['numerical_feature_cols']
    categorical_feature_cols = schema['categorical_feature_cols']
    schema_features = numerical_feature_cols + categorical_feature_cols
    target_col = schema.get('target_column')
    
    # Check for missing required feature columns
    missing_cols = set(schema_features) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Required feature columns missing from DataFrame: {sorted(missing_cols)}")
    
    # Optional downsample for quick tests (coerce max_samples to int if possible)
    if max_samples is None:
        test_subset_df = df
    else:
        try:
            max_n = int(max_samples)
        except Exception:
            max_n = len(df)
        test_subset_df = df.head(min(max_n, len(df)))
    
    # Defer building feature_df until after enforcing feature order and parity filtering

    # Build tensors directly using schema (no target required for inference)
    numerical_feature_cols = schema.get('numerical_feature_cols', [])
    categorical_feature_cols = schema.get('categorical_feature_cols', [])
    scaling_params = schema.get('scaling_params', {})
    apply_scaling = bool(schema.get('apply_scaling', False))
    # Auto-create code twins (c*) strictly from schema.code_mappings if missing (do BEFORE enforcing order)
    code_mappings = schema.get('code_mappings', [])
    if code_mappings:
        for m in code_mappings:
            src = m.get('source')
            code_col = m.get('code_col')
            code_levels = m.get('classes', [])
            mapping = m.get('mapping', {})
            if code_col not in test_subset_df.columns:
                if src not in test_subset_df.columns:
                    raise ValueError(f"Strict mode: required source column '{src}' missing for code twin '{code_col}'")
                vals = test_subset_df[src].to_list()
                codes = []
                for v in vals:
                    if v is None:
                        if 'NULL' in code_levels:
                            codes.append(mapping['NULL'])
                        else:
                            raise ValueError(f"Strict mode: null in '{src}' but no 'NULL' class in schema")
                    else:
                        key = str(v)
                        if key not in mapping:
                            raise ValueError(f"Strict mode: unseen value '{v}' in '{src}'. Allowed: {code_levels}")
                        codes.append(mapping[key])
                test_subset_df = test_subset_df.with_columns(pl.Series(code_col, np.array(codes, dtype=np.int32)))

    # Enforce feature ordering using feature_column_list if present
    feature_column_list = schema.get('feature_column_list')
    if feature_column_list:
        # Extend feature order with any missing code twins so selection doesn't drop them
        if code_mappings:
            missing_codes = [m.get('code_col') for m in code_mappings if m.get('code_col') and m.get('code_col') not in feature_column_list]
            if missing_codes:
                feature_column_list = feature_column_list + missing_codes
        # Reorder DataFrame columns to match training feature order strictly
        missing = [c for c in feature_column_list if c not in test_subset_df.columns]
        if missing:
            raise ValueError(f"Inference DataFrame missing required features: {missing[:10]}{'...' if len(missing)>10 else ''}")
        # Select in exact order
        test_subset_df = test_subset_df.select(feature_column_list)
        # Recompute numeric/categorical lists as positions
        numerical_feature_cols = [c for c in feature_column_list if c in schema.get('numerical_feature_cols', [])]
        categorical_feature_cols = [c for c in feature_column_list if c in schema.get('categorical_feature_cols', [])]
    category_mappings = schema.get('category_mappings', {})
    cat_feature_info = schema.get('cat_feature_info', {})

    # Build feature-only DataFrame in correct order (after enforcing ordering)
    feature_cols_ordered = numerical_feature_cols + categorical_feature_cols
    if target_col and target_col in test_subset_df.columns:
        feature_cols_ordered = feature_cols_ordered + [target_col]
    feature_df = test_subset_df.select(feature_cols_ordered)

    if parity_drop_non_finite and numerical_feature_cols:
        num_np = feature_df.select(numerical_feature_cols).to_numpy()
        finite_mask = np.isfinite(num_np).all(axis=1)
        feature_df = feature_df.filter(pl.Series(name="__mask__", values=finite_mask))

    # Build continuous tensor with schema preprocessing (clip/scale) like classification path
    X_num_np = feature_df.select(numerical_feature_cols).to_numpy().astype(np.float32)
    X_num_np = np.nan_to_num(X_num_np, nan=0.0, posinf=1e6, neginf=-1e6)
    preprocessing = schema.get('preprocessing')
    if preprocessing:
        clip_low, clip_high = preprocessing.get('clip_range', [-100, 100])
        scale_factor = preprocessing.get('scale_factor', 1)
        X_num_np = np.clip(X_num_np, clip_low, clip_high)
        if scale_factor and abs(scale_factor) != 1:
            X_num_np = X_num_np / float(scale_factor)
    elif apply_scaling and scaling_params:
        X_num_np = apply_feature_scaling(X_num_np, numerical_feature_cols, scaling_params, apply_scaling=True)
    Xc = torch.tensor(X_num_np, dtype=torch.float32)
    Xk = build_categorical_tensor(
        feature_df,
        categorical_feature_cols=categorical_feature_cols,
        category_mappings=category_mappings,
        cat_feature_info=cat_feature_info,
    )

    # Prepare cat_dims from schema
    cat_feature_info = schema.get('cat_feature_info', {})
    cat_dims = (
        [(cardinality + 1, min(50, cardinality // 2)) for cardinality in cat_feature_info.values()]
        if cat_feature_info else None
    )

    # Compute expected input size for embeddings pipeline
    embedding_output_dim = sum(d for _, d in (cat_dims or []))
    expected_input_dim = int(Xc.shape[1]) + int(embedding_output_dim)

    # Get true first linear in_features (ignore embedding matrices)
    if 'layers.0.weight' in state_dict and state_dict['layers.0.weight'].ndim == 2:
        in_features = int(state_dict['layers.0.weight'].shape[1])
    else:
        linear_candidates = [
            (k, v) for k, v in state_dict.items()
            if torch.is_tensor(v) and v.ndim == 2 and not k.startswith('embeddings.')
        ]
        if not linear_candidates:
            raise RuntimeError("Could not locate first linear layer in checkpoint")
        in_features = max(linear_candidates, key=lambda kv: kv[1].shape[1])[1].shape[1]

    if in_features != expected_input_dim:
        # Adjust tensors to match checkpoint expectations
        cont_expected = int(in_features - embedding_output_dim)
        # Fix continuous dims by slicing or zero-padding
        if cont_expected != int(Xc.shape[1]):
            if cont_expected < int(Xc.shape[1]):
                Xc = Xc[:, :cont_expected]
            else:
                pad_cols = cont_expected - int(Xc.shape[1])
                pad = torch.zeros((Xc.shape[0], pad_cols), dtype=Xc.dtype)
                Xc = torch.cat([Xc, pad], dim=1)
        # Fix categorical columns count to match number of embeddings
        num_emb_modules = len(cat_dims or [])
        if Xk is not None:
            current_cat_cols = int(Xk.shape[1])
            if num_emb_modules != current_cat_cols:
                if num_emb_modules == 0:
                    Xk = None
                else:
                    if num_emb_modules < current_cat_cols:
                        Xk = Xk[:, :num_emb_modules]
                    else:
                        # pad unknown-index column (zeros) to reach required count
                        pad_cols = num_emb_modules - current_cat_cols
                        pad = torch.zeros((Xk.shape[0], pad_cols), dtype=Xk.dtype)
                        Xk = torch.cat([Xk, pad], dim=1)
        # Recompute expected after adjustments (for sanity only)
        expected_input_dim = int(Xc.shape[1]) + int(embedding_output_dim)

    # Build model: input_dim is continuous-only; embeddings added inside the model
    mlp_layers = schema.get('mlp_layers', [1024, 512, 256, 128])
    dropout = schema.get('mlp_dropout', 0.1)
    
    print(f"DEBUG: Model architecture - input_dim: {int(Xc.shape[1])}, cat_dims: {cat_dims}")
    print(f"DEBUG: MLP layers: {mlp_layers}, dropout: {dropout}")
    
    model = MLP(input_dim=int(Xc.shape[1]), cat_dims=cat_dims, layer_sizes=mlp_layers, dropout=dropout)
    
    # Check model state before loading
    print(f"DEBUG: Model created, loading state dict with {len(state_dict)} parameters")
    
    # Load state dict directly (all models now use layers architecture)
    try:
        model.load_state_dict(state_dict)
        print(f"DEBUG: [OK] State dict loaded successfully")
    except Exception as e:
        print(f"DEBUG: [ERROR] Error loading state dict: {e}")
        print(f"DEBUG: Expected keys: {list(model.state_dict().keys())}")
        print(f"DEBUG: Provided keys: {list(state_dict.keys())}")
        raise

    # Move model to specified device for inference
    model = model.to(device_t).eval()
    
    # Check if model parameters look reasonable
    total_params = sum(p.numel() for p in model.parameters())
    param_stats = []
    for name, param in model.named_parameters():
        if param.numel() > 0:
            param_mean = param.data.mean().item()
            param_std = param.data.std().item()
            param_stats.append(f"{name}: mean={param_mean:.6f}, std={param_std:.6f}")
    
    print(f"DEBUG: Model has {total_params} total parameters")
    print(f"DEBUG: Sample parameter stats: {param_stats[:3]}")
    
    # Check if all parameters are zero (indicating a problem)
    all_zero = all(param.data.abs().max().item() < 1e-8 for param in model.parameters())
    if all_zero:
        print(f"DEBUG: [CRITICAL] All model parameters are near zero!")
    else:
        print(f"DEBUG: [OK] Model parameters have non-zero values")

    # Ensure tensors are correct dtype
    Xc = torch.as_tensor(Xc, dtype=torch.float32)
    Xk = None if Xk is None or (hasattr(Xk, 'numel') and Xk.numel() == 0) else torch.as_tensor(Xk, dtype=torch.int64)

    preds: List[float] = []
    bs = 8192
    
    print(f"DEBUG: Starting inference with {Xc.shape[0]} samples, batch_size={bs}")
    print(f"DEBUG: y_range = {y_range}")
    print(f"DEBUG: Input tensor shapes - Xc: {Xc.shape}, Xk: {Xk.shape if Xk is not None else None}")
    
    # Check input diversity
    if Xc.shape[0] > 1:
        input_std = Xc.std(dim=0).mean().item()
        print(f"DEBUG: Input diversity - mean std across features: {input_std:.6f}")
        if input_std < 1e-6:
            print(f"DEBUG: ⚠️  WARNING: Very low input diversity - inputs may be nearly identical")
    
    temperature = float(schema.get('temperature', 1.0) or 1.0)
    # Apply sigmoid only if explicitly requested by schema
    apply_sigmoid = (schema.get('regression_activation', '').lower() == 'sigmoid')
    try:
        print(f"DEBUG[REG] Inference flags: apply_sigmoid={apply_sigmoid}, schema.temperature={temperature}")
    except Exception:
        pass

    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, Xc.shape[0], bs)):
            xb_cont = Xc[i:i+bs].to(device_t, non_blocking=True)
            xb_cat = None if Xk is None else Xk[i:i+bs].to(device_t, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=(device_t.type == 'cuda')):
                # Get raw model output
                raw_output = model(xb_cont, xb_cat).squeeze(-1)
                
                # Debug first batch
                if batch_idx == 0:
                    raw_vals = raw_output.cpu().tolist()
                    # Handle case where raw_vals is a single float (batch size 1)
                    if isinstance(raw_vals, float):
                        raw_vals = [raw_vals]
                    print(f"DEBUG: Batch {batch_idx} raw model output:")
                    print(f"  Min: {min(raw_vals):.6f}, Max: {max(raw_vals):.6f}")
                    print(f"  Mean: {sum(raw_vals)/len(raw_vals):.6f}")
                    print(f"  Sample: {[f'{x:.6f}' for x in raw_vals[:5]]}")
                    
                    # Check if model is actually computing different values
                    unique_raw = len(set(raw_vals))
                    if unique_raw == 1:
                        print(f"  [CRITICAL] Model outputs identical values: {raw_vals[0]}")
                        print(f"  Model may be broken, not loaded properly, or inputs are identical")
                    else:
                        print(f"  [OK] Model outputs {unique_raw} unique values")
                
                # Map regression output
                if apply_sigmoid:
                    yb = torch.sigmoid(raw_output / temperature)
                    if y_range is not None:
                        y_min, y_max = y_range
                        yb = yb * (y_max - y_min) + y_min
                    if batch_idx == 0:
                        vals = yb.detach().cpu().tolist()
                        if isinstance(vals, float):
                            vals = [vals]
                        print(f"DEBUG: Regression output stats (sigmoid-mapped):")
                        print(f"  Min: {min(vals):.6f}, Max: {max(vals):.6f}")
                        print(f"  Sample: {[f'{x:.6f}' for x in vals[:5]]}")
                else:
                    yb = raw_output
                    if y_range is not None:
                        yb = yb.clamp(min=y_range[0], max=y_range[1])
                    if batch_idx == 0:
                        vals = yb.detach().cpu().tolist()
                        if isinstance(vals, float):
                            vals = [vals]
                        print(f"DEBUG: Regression output stats:")
                        print(f"  Min: {min(vals):.6f}, Max: {max(vals):.6f}")
                        print(f"  Sample: {[f'{x:.6f}' for x in vals[:5]]}")
                    
            batch_preds = yb.cpu().tolist()
            # Handle case where batch_preds is a single float (batch size 1)
            if isinstance(batch_preds, float):
                batch_preds = [batch_preds]
            preds.extend(batch_preds)

    # Optional recentering: previously enforced mean ~0.5 for probability targets.
    # Disabled by default to preserve model calibration; enable via schema['recenter_outputs']=True if desired.
    recenter_outputs = bool(schema.get('recenter_outputs', False))
    recenter_target_std = schema.get('recenter_target_std', None)
    recenter_temperature = float(schema.get('recenter_temperature', 1.0) or 1.0)
    try:
        print(f"DEBUG[REG] Recentering flags: recenter_outputs={recenter_outputs}, target_std={recenter_target_std}, T_init={recenter_temperature}")
    except Exception:
        pass
    try:
        if recenter_outputs and apply_sigmoid and preds:
            p = np.array(preds, dtype=float)
            pre_std = float(np.std(p))
            pre_unique = int(len(np.unique(p)))
            print(f"DEBUG[REG] Pre-recenter stats: mean={float(np.mean(p)):.6f}, std={pre_std:.6f}, unique={pre_unique}")
            if pre_std < 1e-6:
                print(f"DEBUG[REG] Skipping recentering due to near-constant predictions (std<{1e-6})")
            else:
                eps = 1e-6
                p = np.clip(p, eps, 1.0 - eps)
                l = np.log(p / (1.0 - p))

                def _recenter_with_temperature(T: float):
                    lo_c, hi_c = -20.0, 20.0
                    for _ in range(40):
                        c_try = (lo_c + hi_c) / 2.0
                        m = float((1.0 / (1.0 + np.exp(-((l - c_try) / max(T, 1e-6))))).mean())
                        if m > 0.5:
                            lo_c = c_try
                        else:
                            hi_c = c_try
                    c_final = (lo_c + hi_c) / 2.0
                    p_adj = 1.0 / (1.0 + np.exp(-((l - c_final) / max(T, 1e-6))))
                    return p_adj, c_final

                # If target std provided, search T to match it
                if recenter_target_std is not None:
                    try:
                        target_std = float(recenter_target_std)
                        lo_T, hi_T = 0.1, 10.0
                        best_T, best_diff, best_p = recenter_temperature, float('inf'), None
                        for _ in range(30):
                            mid_T = (lo_T + hi_T) / 2.0
                            p_try, c_try = _recenter_with_temperature(mid_T)
                            s = float(np.std(p_try))
                            diff = abs(s - target_std)
                            if diff < best_diff:
                                best_diff, best_T, best_p = diff, mid_T, p_try
                            if s > target_std:
                                lo_T = mid_T
                            else:
                                hi_T = mid_T
                        preds = (best_p if best_p is not None else _recenter_with_temperature(recenter_temperature)[0]).tolist()
                        print(f"DEBUG[REG] Recentered with target_std={target_std:.6f}; T≈{best_T:.4f}, std={float(np.std(preds)):.6f}")
                    except Exception:
                        p_new, c = _recenter_with_temperature(recenter_temperature)
                        preds = p_new.tolist()
                        print(f"DEBUG[REG] Recentered (fallback) with T={recenter_temperature:.4f}; std={float(np.std(p_new)):.6f}")
                else:
                    p_new, c = _recenter_with_temperature(recenter_temperature)
                    preds = p_new.tolist()
                    post_std = float(np.std(p_new))
                    post_unique = int(len(np.unique(p_new)))
                    print(f"DEBUG[REG] Recentered to mean≈{float(np.mean(p_new)):.4f} using T={recenter_temperature:.4f}; std={post_std:.6f}, unique={post_unique}")
        elif apply_sigmoid and preds:
            # Log that recentering is intentionally disabled
            try:
                p = np.array(preds, dtype=float)
                print(f"DEBUG[REG] Recenter disabled. Mean={float(np.mean(p)):.6f}, std={float(np.std(p)):.6f}, unique={int(len(np.unique(p)))}")
            except Exception:
                pass
    except Exception:
        pass

    # Assemble results: return ONLY prediction-related columns
    try:
        arr = np.array(preds, dtype=float)
        if arr.size:
            print(f"DEBUG[REG] Final preds stats: mean={float(np.mean(arr)):.6f}, std={float(np.std(arr)):.6f}, min={float(np.min(arr)):.6f}, max={float(np.max(arr)):.6f}, unique={int(len(np.unique(arr)))}")
    except Exception:
        pass
    out_cols = {
        f'{y_name}_Pred': preds,
    }
    if y_name in test_subset_df.columns:
        actual_values = test_subset_df[y_name].to_numpy()
        errors = actual_values - np.array(preds)
        out_cols[f'{y_name}_Pred_Error'] = errors.tolist()
        out_cols[f'{y_name}_Pred_Absolute_Error'] = np.abs(errors).tolist()

    return pl.DataFrame(out_cols)


def predict_model(
    saved_models_path: pathlib.Path,
    model_name: str,
    df: pl.DataFrame,
    # 🔍 INFERENCE-ONLY PARAMETERS: Do not affect training
    device: str = 'cpu',  # Device for inference: 'cpu' or 'cuda'
    max_samples: Optional[int] = None,
    top_k: int = 1,  # Classification only
    return_probs: bool = False,  # Classification only
    strict_validation: bool = True,  # Strict schema validation
    parity_drop_non_finite: bool = False,  # Mirror training: drop rows with non-finite features
) -> pl.DataFrame:
    """
    Make predictions using trained model and schema parameters.
    
    🔧 SHARED PARAMETERS come from schema (set during schema generation):
    - layers, dropout, y_range, apply_scaling, features, scaling, mappings
    
    🔍 INFERENCE-ONLY PARAMETERS (do not affect training):
    - device: Device for inference ('cpu' or 'cuda'). Defaults to 'cpu' for compatibility.
    - max_samples: Limit number of samples for quick testing
    - top_k: Return top K class predictions (classification only)
    - return_probs: Include prediction probabilities (classification only)
    """
    # Post-training inference: rely on schema only for model_type
    model_type = get_model_type_from_schema(saved_models_path=saved_models_path, model_name=model_name)
    if model_type == 'regression':
        return predict_regression_model(
            saved_models_path=saved_models_path,
            model_name=model_name,
            df=df,
            device=device,
            max_samples=max_samples,
            strict_validation=strict_validation,
            parity_drop_non_finite=parity_drop_non_finite,
        )
    else:
        return predict_classification_model(
            saved_models_path=saved_models_path,
            model_name=model_name,
            df=df,
            device=device,
            top_k=top_k,
            max_samples=max_samples,
            return_probs=return_probs,
            strict_validation=strict_validation,
            parity_drop_non_finite=parity_drop_non_finite,
        )

def predict_classification_model(
    saved_models_path: pathlib.Path,
    model_name: str,
    df: pl.DataFrame,
    classes: Optional[List[Any]] = None,
    idx_to_class: Optional[Dict[int, Any]] = None,
    device: str = 'cpu',
    top_k: int = 1,
    max_samples: Optional[int] = None,
    return_probs: bool = False,
    strict_validation: bool = True,
    parity_drop_non_finite: bool = False,
) -> pl.DataFrame:
    """
    Embeddings-aware classification prediction with softmax and optional top-k outputs.

    Behavior:
    - Builds (X_cont, X_cat) from the schema using the same scaling and categorical
      mappings as training (without requiring target in the input DataFrame).
    - Reconstructs a classification MLP that mirrors training architecture (continuous +
      optional categorical embeddings), using schema's layer sizes and dropout.
    - Loads weights, runs batched inference, and returns predicted codes/labels and
      optional probabilities or top-k predictions.

    Notes on label mapping:
    - If `idx_to_class` is provided, it is used as the definitive mapping.
    - Else if `classes` is provided, indices map to those in order.
    - Else we infer `num_classes` from the checkpoint's last linear layer; labels will
      be integer codes unless `y_name` is present in df and we can derive a vocab.
    """

    # Load model state
    model_file = resolve_model_path(saved_models_path, model_name)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    # Determine device and load model accordingly
    device_t = torch.device(device if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_file, map_location=device_t, weights_only=False)

    # Load schema
    schema_path = resolve_schema_path(saved_models_path, model_name)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    schema = json.load(open(schema_path))
    
    # Validate inference DataFrame dtypes against schema
    df = validate_inference_dataframe_dtypes(df, schema, strict=strict_validation, verbose=True)

    # y_name from schema
    y_name = schema.get('target_column') or 'Target'

    # Optional downsample for quick tests
    test_subset_df = df if max_samples is None else df.head(min(max_samples, len(df)))

    # Build tensors using shared helpers (do not require target column)
    numerical_feature_cols = schema.get('numerical_feature_cols', [])
    categorical_feature_cols = schema.get('categorical_feature_cols', [])
    # Ensure numeric feature list includes any c* twins from schema
    code_mappings = schema.get('code_mappings', [])
    if code_mappings:
        for m in code_mappings:
            cc = m.get('code_col')
            if cc and cc not in numerical_feature_cols:
                numerical_feature_cols.append(cc)
    # Strictly create any required code twins (c*) from schema code_mappings
    code_mappings = schema.get('code_mappings', [])
    if code_mappings:
        for m in code_mappings:
            src = m.get('source')
            code_col = m.get('code_col')
            # Use distinct name to avoid any chance of shadowing target classes
            code_levels = m.get('classes', [])
            mapping = m.get('mapping', {})
            if code_col and code_col not in test_subset_df.columns:
                if src not in test_subset_df.columns:
                    raise ValueError(f"Strict mode: required source column '{src}' missing for code twin '{code_col}'")
                vals = test_subset_df[src].to_list()
                codes = []
                for v in vals:
                    if v is None:
                        if 'NULL' in code_levels:
                            codes.append(mapping['NULL'])
                        else:
                            raise ValueError(f"Strict mode: null in '{src}' but no 'NULL' class in schema")
                    else:
                        key = str(v)
                        if key not in mapping:
                            raise ValueError(f"Strict mode: unseen value '{v}' in '{src}'. Allowed: {code_levels}")
                        codes.append(mapping[key])
                test_subset_df = test_subset_df.with_columns(pl.Series(code_col, np.array(codes, dtype=np.int32)))
                if code_col not in numerical_feature_cols:
                    numerical_feature_cols.append(code_col)

    # Enforce feature order via feature_column_list; fallback to training stats 'feature_cols' if present
    feature_column_list = schema.get('feature_column_list') or schema.get('feature_cols')
    if feature_column_list:
        # Extend order with any c* twins not present so selection won't drop them
        if code_mappings:
            missing_codes = [m.get('code_col') for m in code_mappings if m.get('code_col') and m.get('code_col') not in feature_column_list]
            if missing_codes:
                feature_column_list = feature_column_list + missing_codes
        missing = [c for c in feature_column_list if c not in test_subset_df.columns]
        if missing:
            raise ValueError(f"Inference DataFrame missing required features: {missing[:10]}{'...' if len(missing)>10 else ''}")
        test_subset_df = test_subset_df.select(feature_column_list)
        # Recalculate numeric/categorical lists from ordered columns, but make sure c* twins are included
        base_numeric = set(schema.get('numerical_feature_cols', []))
        if code_mappings:
            for m in code_mappings:
                cc = m.get('code_col')
                if cc:
                    base_numeric.add(cc)
        numerical_feature_cols = [c for c in feature_column_list if c in base_numeric]
        categorical_feature_cols = [c for c in feature_column_list if c in schema.get('categorical_feature_cols', [])]
        try:
            print(f"DEBUG[CLS] Using feature order from {'feature_column_list' if 'feature_column_list' in schema else 'feature_cols'} (n={len(feature_column_list)})")
        except Exception:
            pass
    scaling_params = schema.get('scaling_params', {})
    apply_scaling = bool(schema.get('apply_scaling', False))
    category_mappings = schema.get('category_mappings', {})
    cat_feature_info = schema.get('cat_feature_info', {})

    # If parity mode is enabled, drop rows with non-finite numerical features
    if parity_drop_non_finite and numerical_feature_cols:
        num_np = test_subset_df.select(numerical_feature_cols).to_numpy()
        finite_mask = np.isfinite(num_np).all(axis=1)
        test_subset_df = test_subset_df.filter(pl.Series(name="__mask__", values=finite_mask))

    # Build continuous tensor applying training-time preprocessing when present
    # DEBUG: Check actual DataFrame columns vs what we're trying to select
    print(f"   DEBUG: test_subset_df has {len(test_subset_df.columns)} columns")
    print(f"   DEBUG: Attempting to select {len(numerical_feature_cols)} numerical columns")
    numerical_in_df = [c for c in numerical_feature_cols if c in test_subset_df.columns]
    numerical_missing = [c for c in numerical_feature_cols if c not in test_subset_df.columns]
    if numerical_missing:
        print(f"   DEBUG: Missing numerical columns: {numerical_missing[:10]}")
    print(f"   DEBUG: Found {len(numerical_in_df)} numerical columns in DataFrame")
    
    X_num = test_subset_df.select(numerical_feature_cols).to_numpy().astype(np.float32)
    X_num = np.nan_to_num(X_num, nan=0.0, posinf=1e6, neginf=-1e6)
    preprocessing = schema.get('preprocessing')
    if preprocessing:
        clip_low, clip_high = preprocessing.get('clip_range', [-100, 100])
        scale_factor = preprocessing.get('scale_factor', 1)
        X_num = np.clip(X_num, clip_low, clip_high)
        if scale_factor and abs(scale_factor) != 1:
            X_num = X_num / float(scale_factor)
    elif apply_scaling and scaling_params:
        # Backward compatibility: mean/std scaling path
        X_num = apply_feature_scaling(X_num, numerical_feature_cols, scaling_params, apply_scaling=True)
    # DEBUG: numeric tensor stats for classification
    try:
        print(f"DEBUG[CLS] Numeric tensor stats (n={X_num.shape[0]}): mean={np.mean(X_num):.6f}, std={np.std(X_num):.6f}, min={np.min(X_num):.6f}, max={np.max(X_num):.6f}")
    except Exception:
        pass
    Xc = torch.tensor(X_num, dtype=torch.float32)
    Xk = build_categorical_tensor(
        test_subset_df,
        categorical_feature_cols=categorical_feature_cols,
        category_mappings=category_mappings,
        cat_feature_info=cat_feature_info,
    )

    # Check if model uses embeddings or direct concatenation
    embedding_weight_keys = [k for k in state_dict.keys() if k.startswith('embeddings.') and k.endswith('.weight')]
    uses_embeddings = len(embedding_weight_keys) > 0
    
    if uses_embeddings:
        # Embeddings approach: derive cat_dims from checkpoint
        def _emb_idx(name: str) -> int:
            try:
                return int(name.split('.')[1])
            except Exception:
                return 0
        embedding_weight_keys.sort(key=_emb_idx)
        cat_dims = []
        for k in embedding_weight_keys:
            w = state_dict[k]
            if torch.is_tensor(w) and w.ndim == 2:
                num_embeddings = int(w.shape[0])
                emb_dim = int(w.shape[1])
                cat_dims.append((num_embeddings, emb_dim))
        embedding_output_dim = sum(d for _, d in cat_dims)
        expected_input_dim = int(Xc.shape[1]) + int(embedding_output_dim)
    else:
        # Direct concatenation approach: cat features added as-is
        cat_dims = None
        cat_input_dim = int(Xk.shape[1]) if (Xk is not None and Xk.numel() > 0) else 0
        expected_input_dim = int(Xc.shape[1]) + cat_input_dim
        embedding_output_dim = cat_input_dim

    # Infer first linear in_features (ignore embedding matrices)
    linear_candidates = [
        (k, v) for k, v in state_dict.items()
        if torch.is_tensor(v) and v.ndim == 2 and not k.startswith('embeddings.')
    ]
    if not linear_candidates:
        raise RuntimeError("Could not locate any linear layer weights in checkpoint")

    # Prefer a layer named layers.0.weight when present, else widest input weight
    in_features = None
    if 'layers.0.weight' in state_dict and state_dict['layers.0.weight'].ndim == 2:
        in_features = int(state_dict['layers.0.weight'].shape[1])
    else:
        in_features = max(linear_candidates, key=lambda kv: kv[1].shape[1])[1].shape[1]

    print(f"   Categorical features: {len(categorical_feature_cols)} columns")
    print(f"   Numerical features: {len(numerical_feature_cols)} columns")
    
    # DEBUG: Check for overlap between numerical and categorical
    overlap = set(numerical_feature_cols) & set(categorical_feature_cols)
    if overlap:
        print(f"   ⚠️  OVERLAP detected: {overlap} appears in both numerical and categorical lists")
    
    # DEBUG: Show the calculation breakdown
    print(f"   DEBUG: Xc.shape[1]={int(Xc.shape[1])}, embedding_output_dim={embedding_output_dim}, expected_input_dim={expected_input_dim}")
    print(f"   DEBUG: Model in_features={in_features}")
    
    if in_features != expected_input_dim:
        approach_str = "embeddings" if uses_embeddings else "concatenation"
        print(f"⚠️  Feature count mismatch detected:")
        print(f"   Training schema had: {len(schema.get('numerical_feature_cols', []))} numerical + {len(schema.get('categorical_feature_cols', []))} categorical features")
        print(f"   Inference data has: {len(test_subset_df.columns)} total columns")
        
        # Skip expensive column analysis for large DataFrames (performance optimization)
        if len(test_subset_df.columns) > 1000:
            print("   ⚡ Skipping detailed column analysis (large DataFrame optimization)")
        else:
            # Check if inference data has extra columns not in schema
            numerical_cols = schema.get('numerical_feature_cols', [])
            categorical_cols = schema.get('categorical_feature_cols', [])
            schema_features = set(numerical_cols + categorical_cols)
            inference_features = set(test_subset_df.columns)
            extra_features = inference_features - schema_features
            missing_features = schema_features - inference_features
            
            if extra_features:
                print(f"   Extra columns in inference data (not in training): {sorted(list(extra_features))}")
            if missing_features:
                print(f"   Missing columns from inference data (were in training): {sorted(list(missing_features))}")
            
        raise RuntimeError(
            f"Feature mismatch: Model expects {in_features} inputs but schema computes {expected_input_dim}. "
            f"This suggests the inference DataFrame has different columns than the training data. "
            f"Check that df_test has the same feature columns as the training data used to create the schema."
        )

    # Infer number of classes
    # STRICT: derive number of classes from the checkpoint final linear layer
    layers_weight_layers = [(k, v) for k, v in linear_candidates if k.startswith('layers.') and k.endswith('.weight')]
    def _layers_index(name: str) -> int:
        m = re.match(r'^layers\.(\d+)\.weight$', name)
        return int(m.group(1)) if m else -1
    if layers_weight_layers:
        last_key, last_w = max(layers_weight_layers, key=lambda kv: _layers_index(kv[0]))
        num_classes = int(last_w.shape[0])
    else:
        num_classes = int(min(linear_candidates, key=lambda kv: kv[1].shape[0])[1].shape[0])

    # Build a classification MLP with embeddings matching checkpoint shapes (if any)
    class MLPClassification(torch.nn.Module):
        def __init__(self, dim_in: int, cat_dims_local: Optional[List[Tuple[int, int]]], widths: List[int], dropout: float, out_classes: int):
            super().__init__()
            self.embeddings = torch.nn.ModuleList()
            embedding_output_dim_local = 0
            if cat_dims_local:
                for c, emb_dim in cat_dims_local:
                    self.embeddings.append(torch.nn.Embedding(c, emb_dim))
                    embedding_output_dim_local += emb_dim
            # For concatenation approach, dim_in already includes categorical dimensions
            total_in = dim_in if cat_dims_local is None else dim_in + embedding_output_dim_local
            blocks = []
            prev = total_in
            for w in widths:
                blocks.append(torch.nn.Linear(prev, w))
                blocks.append(torch.nn.ReLU())
                if dropout and dropout > 0:
                    blocks.append(torch.nn.Dropout(dropout))
                prev = w
            blocks.append(torch.nn.Linear(prev, out_classes))
            self.layers = torch.nn.Sequential(*blocks)
        def forward(self, x_cont, x_cat=None):
            if self.embeddings and x_cat is not None and x_cat.numel() > 0:
                # Embeddings approach
                embedded = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.embeddings)]
                x = torch.cat([x_cont, torch.cat(embedded, 1)], 1)
            elif x_cat is not None and x_cat.numel() > 0:
                # Direct concatenation approach (like training)
                x = torch.cat([x_cont, x_cat.float()], 1)
            else:
                x = x_cont
            return self.layers(x)

    mlp_layers = schema.get('mlp_layers', [1024, 512, 256, 128])
    dropout = schema.get('mlp_dropout', 0.1)
    
    if uses_embeddings:
        # Embeddings approach: continuous input + embedding dimensions
        model = MLPClassification(dim_in=int(Xc.shape[1]), cat_dims_local=cat_dims, widths=mlp_layers, dropout=dropout, out_classes=num_classes)
    else:
        # Direct concatenation approach: continuous input + categorical input
        cont_dim = int(Xc.shape[1])
        cat_dim = int(Xk.shape[1]) if (Xk is not None and Xk.numel() > 0) else 0
        total_input_dim = cont_dim + cat_dim
        model = MLPClassification(dim_in=total_input_dim, cat_dims_local=None, widths=mlp_layers, dropout=dropout, out_classes=num_classes)

    # Load weights strictly to catch any silent shape/key drift
    model.load_state_dict(state_dict, strict=True)

    # Move model to specified device for inference
    model = model.to(device_t).eval()

    # Ensure tensor dtypes and existence
    Xc = torch.as_tensor(Xc, dtype=torch.float32)
    if uses_embeddings:
        Xk = None if Xk is None or (hasattr(Xk, 'numel') and Xk.numel() == 0) else torch.as_tensor(Xk, dtype=torch.int64)
    else:
        # For concatenation approach, categorical features should be float
        Xk = None if Xk is None or (hasattr(Xk, 'numel') and Xk.numel() == 0) else torch.as_tensor(Xk, dtype=torch.float32)

    pred_codes: List[int] = []
    topk_codes: List[List[int]] = []
    probs_out: List[List[float]] = []

    bs = 8192
    temperature = float(schema.get('temperature', 1.0) or 1.0)
    with torch.no_grad():
        max_prob_values: List[float] = []
        for i in range(0, Xc.shape[0], bs):
            xb_cont = Xc[i:i+bs].to(device_t, non_blocking=True)
            xb_cat = None if Xk is None else Xk[i:i+bs].to(device_t, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device_t.type == 'cuda')):
                if uses_embeddings:
                    # Embeddings approach: pass continuous and categorical separately
                    logits = model(xb_cont, xb_cat)
                else:
                    # Concatenation approach: combine inputs in forward method
                    logits = model(xb_cont, xb_cat)
                # Temperature scaling for calibration (if provided in schema)
                if temperature and temperature != 1.0:
                    logits = logits / float(temperature)
                batch_probs = F.softmax(logits, dim=-1)
                batch_pred = torch.argmax(batch_probs, dim=-1)
            batch_pred_list = batch_pred.cpu().tolist()
            pred_codes.extend(batch_pred_list)
            
            # Debug: Show raw model outputs for first few predictions
            if len(pred_codes) <= 5:  # Only for first batch
                batch_probs_list = batch_probs.cpu().tolist()
                for j in range(min(5, len(batch_pred_list))):
                    pred_idx = batch_pred_list[j]
                    probs = batch_probs_list[j]
                    top_5_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:5]
                    print(f"DEBUG: Sample {len(pred_codes)-len(batch_pred_list)+j}: predicted_idx={pred_idx}, top_5_probs={top_5_probs}")
            
            # Collect max-probability for diagnostics
            try:
                max_prob_values.extend(torch.max(batch_probs, dim=1).values.cpu().tolist())
            except Exception:
                pass
            if return_probs:
                probs_out.extend(batch_probs.cpu().tolist())
            if top_k and top_k > 1:
                topk = torch.topk(batch_probs, k=min(top_k, batch_probs.shape[1]), dim=-1)
                topk_codes.extend(topk.indices.cpu().tolist())

    # Print prediction confidence summary
    try:
        if max_prob_values:
            mp = np.array(max_prob_values, dtype=float)
            print(f"DEBUG[CLS] Confidence stats: mean={mp.mean():.4f}, std={mp.std():.4f}, min={mp.min():.4f}, max={mp.max():.4f}")
    except Exception:
        pass

    # Build label mapping
    if idx_to_class is None and classes is not None:
        idx_to_class = {i: c for i, c in enumerate(classes)}
    if idx_to_class is None:
        # Prefer explicit class_to_idx from schema (comes from training), else fall back
        class_to_idx_map = schema.get('class_to_idx', {})
        if class_to_idx_map:
            idx_to_class = {v: k for k, v in class_to_idx_map.items()}
        else:
            # Then try label_mapping format (string index keys)
            label_mapping = schema.get('label_mapping', {})
            if label_mapping:
                idx_to_class = {int(k): v for k, v in label_mapping.items()}
            else:
                classes_list = schema.get('classes', [])
                if classes_list:
                    idx_to_class = {i: c for i, c in enumerate(classes_list)}

    # DEBUG: Show mapping and predicted distribution for diagnostics
    try:
        if idx_to_class is not None:
            sample_keys = sorted(list(idx_to_class.keys()))[:20]
            sample_map = {k: idx_to_class[k] for k in sample_keys}
            print(f"DEBUG[CLS] idx_to_class sample (first 20): {sample_map}")
        # Predicted code counts top 20
        if pred_codes:
            import collections as _collections
            cnt = _collections.Counter(pred_codes)
            top20 = cnt.most_common(20)
            if idx_to_class is not None:
                top20_lbl = [(idx_to_class.get(k, k), v) for k, v in top20]
                print(f"DEBUG[CLS] Top-20 predicted labels: {top20_lbl}")
            else:
                print(f"DEBUG[CLS] Top-20 predicted codes: {top20}")
        # Compare mapping labels to schema classes if present
        classes_list_dbg = schema.get('classes', [])
        if classes_list_dbg and idx_to_class is not None:
            mapping_labels = set(idx_to_class.values())
            diff_schema_minus_map = set(classes_list_dbg) - mapping_labels
            diff_map_minus_schema = mapping_labels - set(classes_list_dbg)
            print(f"DEBUG[CLS] Mapping vs schema classes: missing_in_map={len(diff_schema_minus_map)}, extra_in_map={len(diff_map_minus_schema)}")
    except Exception:
        pass

    # Assemble output DataFrame
    base_name = y_name if (y_name is not None and len(str(y_name)) > 0) else 'Target'
    # Build output with ONLY prediction columns
    out_cols: Dict[str, Any] = {}
    # Predicted labels (class names) - prioritize over codes
    if idx_to_class is not None:
        pred_labels = [idx_to_class.get(i, i) for i in pred_codes]
        out_cols[f'{base_name}_Pred'] = pred_labels
    else:
        out_cols[f'{base_name}_Pred'] = pred_codes

    # Optional probabilities
    if return_probs:
        out_cols[f'{base_name}_Probs'] = probs_out

    # Optional top-k
    if top_k and top_k > 1:
        out_cols[f'{base_name}_Top{top_k}_Pred_Codes'] = topk_codes
        if idx_to_class is not None:
            topk_labels = [[idx_to_class.get(i, i) for i in row] for row in topk_codes]
            out_cols[f'{base_name}_Top{top_k}_Preds'] = topk_labels

    # If actual labels present, compute matches and errors
    if y_name is not None and y_name in test_subset_df.columns:
        # Align predicted labels to share categorical codes with y_name when possible
        try:
            pass  # output only contains prediction columns now
        except Exception:
            pass

        try:
            actual_vals = test_subset_df.select([y_name]).to_numpy().reshape(-1)

            # Label matching (compare names if available)
            try:
                pred_labels_arr = pl.Series(out_cols[f'{base_name}_Pred']).to_numpy().reshape(-1)
                label_match = [bool(p == a) for p, a in zip(pred_labels_arr, actual_vals)]
                result_df = result_df.with_columns([
                    pl.Series(f'{base_name}_Label_Match', label_match)
                ])
            except Exception:
                pass

            # Numeric error calculation using codes derived from mapping or categorical physical codes
            pred_codes_arr = None
            actual_codes_arr = None

            # Prefer mapping from idx_to_class if present
            if idx_to_class is not None:
                class_to_idx_map = {v: k for k, v in idx_to_class.items()}
                # Map predicted labels to codes
                try:
                    pred_codes_arr = []
                    for v in pred_labels_arr:
                        pred_codes_arr.append(class_to_idx_map.get(v, float('nan')))
                except Exception:
                    pred_codes_arr = None
                # Map actual labels to codes
                try:
                    actual_codes_arr = []
                    for v in actual_vals:
                        actual_codes_arr.append(class_to_idx_map.get(v, float('nan')))
                except Exception:
                    actual_codes_arr = None

            # Fallback: use categorical physical codes from Polars
            if pred_codes_arr is None:
                try:
                    pred_codes_arr = pl.Series(out_cols[f'{base_name}_Pred']).cast(pl.Categorical).to_physical().to_numpy().reshape(-1)
                except Exception:
                    pred_codes_arr = None
            if actual_codes_arr is None:
                try:
                    actual_codes_arr = result_df.select([
                        pl.col(y_name).cast(pl.Categorical).to_physical().alias('c')
                    ]).to_numpy().reshape(-1)
                except Exception:
                    actual_codes_arr = None

            if pred_codes_arr is not None and actual_codes_arr is not None:
                errors = []
                absolute_errors = []
                for a, p in zip(actual_codes_arr, pred_codes_arr):
                    try:
                        af = float(a)
                        pf = float(p)
                        if np.isfinite(af) and np.isfinite(pf):
                            err = af - pf
                            errors.append(err)
                            absolute_errors.append(abs(err))
                        else:
                            errors.append(0.0)
                            absolute_errors.append(0.0)
                    except Exception:
                        errors.append(0.0)
                        absolute_errors.append(0.0)

                out_cols[f'{base_name}_Pred_Error'] = errors
                out_cols[f'{base_name}_Pred_Absolute_Error'] = absolute_errors

        except Exception:
            pass

    return pl.DataFrame(out_cols)


# gpt-5 says: Consider adding calibration (reliability) curve and binned MAE plots.
def _create_prediction_plots(y_name, preds, actuals, errors, analysis_results, plot_sample_size):
    """
    Helper function to create prediction analysis plots.
    FIXED: Memory-efficient sampling and improved error handling.
    """
    plt.figure(figsize=(16, 12))

    # FIXED: Memory-efficient sampling - use streaming approach for large arrays
    if len(preds) > plot_sample_size:
        # Use step-based sampling instead of random.choice to avoid creating large index arrays
        step = max(1, len(preds) // plot_sample_size)
        indices = slice(0, len(preds), step)
        plot_preds = preds[indices]
        plot_actuals = actuals[indices]
        plot_errors = errors[indices]
        # Trim to exact size if needed
        if len(plot_preds) > plot_sample_size:
            plot_preds = plot_preds[:plot_sample_size]
            plot_actuals = plot_actuals[:plot_sample_size]
            plot_errors = plot_errors[:plot_sample_size]
    else:
        plot_preds = preds
        plot_actuals = actuals
        plot_errors = errors

    # 1. Predictions vs Actuals
    plt.subplot(2, 3, 1)
    plt.scatter(plot_actuals, plot_preds, alpha=0.5, s=6)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.xlabel(f'Actual {y_name}')
    plt.ylabel(f'Predicted {y_name}')
    plt.title('Predictions vs Actuals')
    plt.legend()

    # 2. Error histogram
    plt.subplot(2, 3, 2)
    plt.hist(plot_errors, bins=50, alpha=0.7)
    plt.xlim(-1, 1)
    plt.xticks(np.arange(-1.0, 1.01, 0.2), rotation=45, ha='right')
    ax = plt.gca()
    ax.set_xlabel('Prediction Error (Actual - Predicted)', labelpad=10)
    ax.tick_params(axis='x', pad=8)
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.axvline(np.mean(plot_errors), color='red', linestyle='--', label=f'Mean: {np.mean(plot_errors):.4f}')
    plt.legend(loc='upper right')

    # 3. Value distributions
    plt.subplot(2, 3, 3)
    plt.hist(plot_preds, bins=50, alpha=0.7, label='Predictions')
    plt.hist(plot_actuals, bins=50, alpha=0.7, label='Actuals')
    plt.xlim(0, 1)
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Value Distributions')
    plt.legend()

    # 4. Residuals vs predicted
    plt.subplot(2, 3, 4)
    plt.scatter(plot_preds, plot_errors, alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residuals vs Predicted')
    plt.axhline(0, color='red', linestyle='--')

    # 5. Data quality overview
    plt.subplot(2, 3, 5)
    categories = ['Valid', 'NaN Pred', 'NaN Actual']
    counts = [
        analysis_results['valid_samples'],
        analysis_results['nan_predictions'],
        analysis_results['nan_actuals']
    ]
    colors = ['green', 'red', 'orange']
    plt.bar(categories, counts, color=colors, alpha=0.7)
    plt.title('Data Quality Overview')
    plt.ylabel('Count')
    for i, count in enumerate(counts):
        plt.text(i, count + analysis_results['total_samples']*0.01, f'{count:,}', ha='center')

    # 6. Absolute error distribution
    plt.subplot(2, 3, 6)
    abs_errors = np.abs(plot_errors)
    plt.hist(abs_errors, bins=50, alpha=0.7)
    plt.xlim(0, 1)
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Absolute Error Distribution')
    plt.axvline(np.mean(abs_errors), color='red', linestyle='--', label=f'Mean: {np.mean(abs_errors):.4f}')
    plt.legend()

    plt.tight_layout()
    plt.show()


def analyze_prediction_results_regression(
    results_df,
    y_name: str,
    prediction_col_pattern: str = 'Pred',
    create_plots: bool = True,
    plot_sample_size: int = 100000,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive analysis of ML model prediction results.

    Parameters:
    -----------
    results_df : pl.DataFrame or pd.DataFrame
        DataFrame containing actual values and predictions
    target_col : str
        Name of the column containing actual/target values
    prediction_col_pattern : str
        Pattern to identify prediction columns (e.g., 'Pred')
    create_plots : bool
        Whether to create visualization plots
    plot_sample_size : int
        Maximum number of points to use for plotting
    verbose : bool
        Whether to print detailed analysis

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing analysis results and statistics
    """

    if verbose:
        print("=== Model Prediction Analysis ===")

    # Check if DataFrame exists and has data
    if results_df is None:
        if verbose:
            print("❌ No results DataFrame provided")
        return {'error': 'No DataFrame provided'}

    try:
        df_len = len(results_df)
        if df_len == 0:
            if verbose:
                print("❌ Results DataFrame is empty")
            return {'error': 'Empty DataFrame'}
    except Exception as e:
        if verbose:
            print(f"❌ Error checking DataFrame length: {e}")
        return {'error': f'Invalid DataFrame: {e}'}

    if verbose:
        print(f"✅ Found results_df with shape: {results_df.shape}")

    # Get all columns and find prediction-related ones
    all_columns = list(results_df.columns)

    # Find prediction columns
    prediction_cols = [col for col in all_columns 
                      if prediction_col_pattern in col and y_name in col and 'Error' not in col]
    actual_cols = [col for col in all_columns if col == y_name]
    error_cols = [col for col in all_columns if 'Error' in col and y_name in col]

    available_cols = actual_cols + prediction_cols + error_cols

    if verbose:
        print(f"Found prediction-related columns: {available_cols}")
        print(f"Pure prediction columns (no Error): {prediction_cols}")

    # Validate we have the necessary columns
    if len(prediction_cols) == 0 or len(actual_cols) == 0:
        error_msg = f"Insufficient columns for analysis. Prediction cols: {prediction_cols}, Actual cols: {actual_cols}"
        if verbose:
            print(f"❌ {error_msg}")
        return {'error': error_msg}

    # Get the main prediction column
    main_pred_col = prediction_cols[0]

    if verbose:
        print(f"\n🔍 DATA QUALITY CHECK:")
        print(f"Using prediction column: {main_pred_col}")

    # Check DataFrame type
    is_polars = hasattr(results_df, 'select') and not hasattr(results_df, 'loc')

    if verbose:
        print(f"DataFrame type: {'Polars' if is_polars else 'Pandas'}")

    # Initialize results dictionary
    analysis_results = {
        'dataframe_type': 'polars' if is_polars else 'pandas',
        'total_samples': len(results_df),
        'prediction_column': main_pred_col,
        'target_column': y_name
    }

    try:
        if is_polars:
            # Polars DataFrame analysis

            # Check for NaN values
            nan_count_pred = results_df.select(pl.col(main_pred_col).is_null().sum()).item()
            nan_count_actual = results_df.select(pl.col(y_name).is_null().sum()).item()

            analysis_results.update({
                'nan_predictions': nan_count_pred,
                'nan_actuals': nan_count_actual,
                'nan_pred_pct': nan_count_pred / len(results_df) * 100,
                'nan_actual_pct': nan_count_actual / len(results_df) * 100
            })

            if verbose:
                print(f"  NaN values in predictions: {nan_count_pred:,} ({nan_count_pred/len(results_df)*100:.1f}%)")
                print(f"  NaN values in actuals: {nan_count_actual:,} ({nan_count_actual/len(results_df)*100:.1f}%)")

            # Get valid (non-NaN) data
            valid_df = results_df.filter(
                pl.col(main_pred_col).is_not_null() & pl.col(y_name).is_not_null()
            )
            valid_count = len(valid_df)

            analysis_results['valid_samples'] = valid_count
            analysis_results['valid_pct'] = valid_count / len(results_df) * 100

            if verbose:
                print(f"  Valid pairs (both non-NaN): {valid_count:,} ({valid_count/len(results_df)*100:.1f}%)")

            if valid_count > 0:
                # Extract valid data
                preds = valid_df[main_pred_col].to_numpy()
                actuals = valid_df[y_name].to_numpy()

                # Calculate statistics
                pred_stats = {
                    'mean': float(np.mean(preds)),
                    'std': float(np.std(preds)),
                    'min': float(np.min(preds)),
                    'max': float(np.max(preds))
                }

                actual_stats = {
                    'mean': float(np.mean(actuals)),
                    'std': float(np.std(actuals)),
                    'min': float(np.min(actuals)),
                    'max': float(np.max(actuals))
                }

                analysis_results.update({
                    'prediction_stats': pred_stats,
                    'actual_stats': actual_stats
                })

                if verbose:
                    print(f"\n📊 PREDICTION STATISTICS (Valid data only):")
                    print(f"  Predictions - Mean: {pred_stats['mean']:.6f}, Std: {pred_stats['std']:.6f}")
                    print(f"  Actuals     - Mean: {actual_stats['mean']:.6f}, Std: {actual_stats['std']:.6f}")
                    print(f"  Pred Range: {pred_stats['min']:.3f} to {pred_stats['max']:.3f}")

                # Calculate errors - handle None values
                errors = actuals - preds
                
                # Filter out any None or NaN values from errors
                # Convert to float array to handle None values properly
                errors_float = np.array(errors, dtype=float)
                valid_mask = ~np.isnan(errors_float) & np.isfinite(errors_float)
                valid_errors = errors_float[valid_mask]
                abs_errors = np.abs(valid_errors)

                if len(valid_errors) > 0:
                    error_stats = {
                        'mean_error': float(np.mean(valid_errors)),
                        'error_std': float(np.std(valid_errors)),
                        'mae': float(np.mean(abs_errors)),
                        'rmse': float(np.sqrt(np.mean(valid_errors**2))),
                        'over_predictions': int(np.sum(valid_errors < 0)),
                        'under_predictions': int(np.sum(valid_errors > 0)),
                        'perfect_predictions': int(np.sum(abs_errors < 0.01))
                    }
                else:
                    # No valid errors to calculate
                    error_stats = {
                        'mean_error': 0.0,
                        'error_std': 0.0,
                        'mae': 0.0,
                        'rmse': 0.0,
                        'over_predictions': 0,
                        'under_predictions': 0,
                        'perfect_predictions': 0
                    }

                # Calculate percentages based on valid errors
                total_errors = len(valid_errors) if len(valid_errors) > 0 else 1  # Avoid division by zero
                error_stats.update({
                    'over_pred_pct': error_stats['over_predictions'] / total_errors * 100,
                    'under_pred_pct': error_stats['under_predictions'] / total_errors * 100,
                    'perfect_pred_pct': error_stats['perfect_predictions'] / total_errors * 100,
                    'valid_error_count': len(valid_errors),
                    'total_error_count': len(errors)
                })

                analysis_results['error_stats'] = error_stats

                if verbose:
                    print(f"\n📈 ERROR ANALYSIS:")
                    print(f"  Mean error (bias): {error_stats['mean_error']:.6f}")
                    print(f"  Error std:         {error_stats['error_std']:.6f}")
                    print(f"  MAE:               {error_stats['mae']:.6f}")
                    print(f"  RMSE:              {error_stats['rmse']:.6f}")

                    print(f"\n📈 ERROR DISTRIBUTION:")
                    print(f"  Over-predictions (pred > actual): {error_stats['over_predictions']} ({error_stats['over_pred_pct']:.1f}%)")
                    print(f"  Under-predictions (pred < actual): {error_stats['under_predictions']} ({error_stats['under_pred_pct']:.1f}%)")
                    print(f"  Perfect predictions: {error_stats['perfect_predictions']} ({error_stats['perfect_pred_pct']:.1f}%)")

                # Display sample data
                if verbose:
                    print(f"\n📋 SAMPLE PREDICTIONS (Valid data):")
                    display_cols = [y_name, main_pred_col]
                    if error_cols:
                        display_cols.append(error_cols[0])

                    valid_sample = valid_df.select(display_cols).head(5)
                    print("✅ Valid predictions:")
                    display(valid_sample)

                    if nan_count_pred > 0:
                        print("\n❌ Sample with NaN predictions:")
                        nan_sample = results_df.filter(pl.col(main_pred_col).is_null()).select(display_cols).head(5)
                        display(nan_sample)

                # Create visualizations
                if create_plots and valid_count >= 100:
                    try:
                        _create_prediction_plots(y_name, preds, actuals, errors, analysis_results, plot_sample_size)
                    except Exception as plot_error:
                        if verbose:
                            print(f"⚠️  Plotting failed: {plot_error}")
                        analysis_results['plot_error'] = str(plot_error)

                # Generate conclusion
                data_quality_good = valid_count >= len(results_df) * 0.8
                analysis_results['data_quality_good'] = data_quality_good

                if verbose:
                    print(f"\n🎯 CONCLUSION:")
                    if data_quality_good:
                        print(f"✅ The model is working well!")
                        print(f"✅ Good data quality: {valid_count:,}/{len(results_df):,} valid predictions ({valid_count/len(results_df)*100:.1f}%)")
                        print(f"✅ Prediction variance: {pred_stats['std']:.6f}")
                        print(f"✅ Error rates: MAE = {error_stats['mae']:.6f}")
                    else:
                        print(f"⚠️  Model has data quality issues:")
                        print(f"⚠️  Only {valid_count:,}/{len(results_df):,} valid predictions ({valid_count/len(results_df)*100:.1f}%)")
                        print(f"⚠️  {nan_count_pred:,} NaN predictions need investigation")

            else:
                error_msg = "No valid prediction pairs found!"
                if verbose:
                    print(f"❌ {error_msg}")
                analysis_results['error'] = error_msg

        else:
            # Pandas DataFrame support
            error_msg = "Pandas support not implemented in this version - please use Polars DataFrame"
            if verbose:
                print(f"❌ {error_msg}")
            analysis_results['error'] = error_msg

    except Exception as e:
        error_msg = f"Error during analysis: {str(e)}"
        if verbose:
            print(f"❌ {error_msg}")
        analysis_results['error'] = error_msg

    return analysis_results


def analyze_prediction_results(
    results_df: pl.DataFrame,
    y_name: str,
    **kwargs,
) -> Dict[str, Any]:
    """Wrapper: choose regression/classification analysis via shared inference."""
    # Use inference on DataFrame
    model_type = infer_model_type_from_df(df=results_df, y_name=y_name)
    if model_type == 'regression':
        return analyze_prediction_results_regression(
            results_df=results_df,
            y_name=y_name,
            prediction_col_pattern=kwargs.get('prediction_col_pattern', 'Pred'),
            create_plots=kwargs.get('create_plots', True),
            plot_sample_size=kwargs.get('plot_sample_size', 100000),
            verbose=kwargs.get('verbose', True),
        )
    else:
        return analyze_prediction_results_classification(
            results_df=results_df,
            y_name=y_name,
            classes=kwargs.get('classes'),
            create_plots=kwargs.get('create_plots', True),
            verbose=kwargs.get('verbose', True),
        )

def _create_classification_plots(y_true, y_pred, confusion_matrix, labels, per_class_metrics, y_true_col, top_k: Optional[int] = None):
    """
    Helper function to create classification analysis plots.
    """
    
    n_classes = len(labels)
    
    if n_classes <= 20:  # Detailed plots allowed up to 20 classes
        plt.figure(figsize=(16, 12))
        
        # 1. Confusion Matrix Heatmap
        plt.subplot(2, 3, 1)
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, cbar_kws={'shrink': 0.8})
        title_k = f"top-k={top_k}" if (top_k is not None and top_k >= 1) else "top-k=1"
        plt.title(f'Confusion Matrix ({title_k})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 2. Per-Class Precision
        plt.subplot(2, 3, 2)
        classes = [str(item['class']) for item in per_class_metrics]
        precisions = [item['precision'] for item in per_class_metrics]
        bars = plt.bar(range(len(classes)), precisions)
        plt.title('Precision by Class')
        plt.xlabel('Class')
        plt.ylabel('Precision')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Color bars by value
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.RdYlGn(precisions[i]))
        
        # 3. Per-Class Recall
        plt.subplot(2, 3, 3)
        recalls = [item['recall'] for item in per_class_metrics]
        bars = plt.bar(range(len(classes)), recalls)
        plt.title('Recall by Class')
        plt.xlabel('Class')
        plt.ylabel('Recall')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Color bars by value
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.RdYlGn(recalls[i]))
        
        # 4. Per-Class F1 Score
        plt.subplot(2, 3, 4)
        f1_scores = [item['f1'] for item in per_class_metrics]
        bars = plt.bar(range(len(classes)), f1_scores)
        plt.title('F1 Score by Class')
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Color bars by value
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.RdYlGn(f1_scores[i]))
        
        # 5. Class Distribution (True)
        plt.subplot(2, 3, 5)
        true_counts = {}
        for label in y_true:
            if label is not None:
                true_counts[label] = true_counts.get(label, 0) + 1
        
        if true_counts:
            # Use natural encounter order from y_true (dict preserves insertion order)
            classes_dist = list(true_counts.keys())
            counts_dist = [true_counts[c] for c in classes_dist]
            plt.bar(range(len(classes_dist)), counts_dist)
            plt.title('True Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(range(len(classes_dist)), [str(c) for c in classes_dist], rotation=45, ha='right')
        
        # 6. Prediction Accuracy by Class
        plt.subplot(2, 3, 6)
        class_accuracies = []
        for i, label in enumerate(labels):
            # Calculate per-class accuracy as TP / (TP + FN)
            tp = confusion_matrix[i, i]
            total_true = confusion_matrix[i, :].sum()
            accuracy = tp / max(1, total_true)
            class_accuracies.append(accuracy)
        
        bars = plt.bar(range(len(labels)), class_accuracies)
        plt.title('Accuracy by Class')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(labels)), [str(l) for l in labels], rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Color bars by value
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.RdYlGn(class_accuracies[i]))
        
        plt.tight_layout()
        plt.show()
    else:
        # For many classes, show simplified plots as two separate figures
        # Annotate counts; if not enough room, reduce to top-N frequent classes until legible
        # Count frequency of true labels
        true_counts_map = {}
        for v in y_true:
            if v is not None:
                true_counts_map[v] = true_counts_map.get(v, 0) + 1
        # Order labels by frequency (descending), then by string for stability
        sorted_labels = sorted(true_counts_map.items(), key=lambda kv: (-kv[1], str(kv[0])))
        # Try decreasing caps until matrix is small enough to annotate
        candidates = [100, 80, 60, 50, 40, 30, 20, 10]
        plot_labels = list(labels)
        cm_view = confusion_matrix
        label_to_idx_local = {lbl: i for i, lbl in enumerate(labels)}
        for cap in candidates:
            if not sorted_labels:
                break
            sel = [kv[0] for kv in sorted_labels[:min(cap, len(sorted_labels))]]
            sel_idx = [label_to_idx_local[lbl] for lbl in sel if lbl in label_to_idx_local]
            if sel_idx:
                cm_try = confusion_matrix[np.ix_(sel_idx, sel_idx)]
                # Aim for annotatable size (<= 40 classes)
                if cm_try.shape[0] <= 40:
                    cm_view = cm_try
                    plot_labels = sel
                    break
        # 1) Confusion Matrix heatmap (wider and significantly taller), with number annotations
        fig_cm = plt.figure(figsize=(24, 18))
        ax1 = fig_cm.add_subplot(1, 1, 1)
        sns.heatmap(
            cm_view,
            cmap='Blues',
            annot=True,
            fmt='d',
            annot_kws={'size': 7},
            cbar_kws={'shrink': 0.9, 'pad': 0.01, 'aspect': 50, 'fraction': 0.02},  # skinny colorbar
            xticklabels=[str(l) for l in plot_labels],
            yticklabels=[str(l) for l in plot_labels]
        )
        title_k = f"top-k={top_k}" if (top_k is not None and top_k >= 1) else "top-k=1"
        ax1.set_title(f'Confusion Matrix ({title_k}, {n_classes} classes)')
        title_k = f"top-k={top_k}" if (top_k is not None and top_k >= 1) else "top-k=1"
        ax1.set_title(f'Confusion Matrix ({title_k}, {n_classes} classes)')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        plt.tight_layout()
        plt.show()

        # 2) Metrics distribution histogram in its own figure
        fig_metrics = plt.figure(figsize=(22, 8))
        ax2 = fig_metrics.add_subplot(1, 1, 1)
        precisions = [item['precision'] for item in per_class_metrics]
        recalls = [item['recall'] for item in per_class_metrics]
        f1_scores = [item['f1'] for item in per_class_metrics]
        ax2.hist([precisions, recalls, f1_scores], bins=20, alpha=0.7,
                 label=['Precision', 'Recall', 'F1'], color=['red', 'green', 'blue'])
        ax2.set_title('Metrics Distribution')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        plt.tight_layout()
        plt.show()


def analyze_prediction_results_classification(
    results_df: pl.DataFrame,
    y_name: str,
    probs_col: Optional[str] = None,
    topk_preds_col: Optional[str] = None,
    classes: Optional[List[Any]] = None,
    create_plots: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    # Derive column names from y_name
    y_true_col = y_name
    y_pred_col = f"{y_name}_Pred"
    y_true = np.array(results_df.select([y_true_col]).to_numpy().reshape(-1), dtype=object)
    y_pred = np.array(results_df.select([y_pred_col]).to_numpy().reshape(-1), dtype=object)

    # Determine label set and mapping using class code ordering when possible
    labels: List[Any] = []
    code_to_name: Dict[int, Any] = {}
    name_to_code: Dict[Any, int] = {}

    # 1) Preferred: use provided classes as code order
    if classes is not None:
        for i, c in enumerate(classes):
            code_to_name[i] = c
            name_to_code[c] = i
    else:
        # 2) Try to infer from paired Pred_Code/Pred columns
        pred_code_col = f"{y_name}_Pred_Code"
        pred_name_col = f"{y_name}_Pred"
        if pred_code_col in results_df.columns and pred_name_col in results_df.columns:
            codes_arr = results_df.select([pred_code_col]).to_numpy().reshape(-1)
            names_arr = results_df.select([pred_name_col]).to_numpy().reshape(-1)
            for c, n in zip(codes_arr, names_arr):
                try:
                    ci = int(c)
                    if n is not None and ci not in code_to_name:
                        code_to_name[ci] = n
                        name_to_code[n] = ci
                except Exception:
                    continue
        # 3) Try to infer from TopK paired columns if present
        if not code_to_name:
            # Auto-detect a TopK codes column for this y
            detected_codes_col = None
            for col in results_df.columns:
                if col.startswith(f"{y_name}_Top") and col.endswith("_Pred_Codes"):
                    detected_codes_col = col
                    break
            if detected_codes_col is not None:
                detected_names_col = detected_codes_col.replace("_Pred_Codes", "_Preds")
                if detected_names_col in results_df.columns:
                    topk_codes = results_df.select([detected_codes_col]).to_numpy().reshape(-1)
                    topk_names = results_df.select([detected_names_col]).to_numpy().reshape(-1)
                    for codes_list, names_list in zip(topk_codes, topk_names):
                        try:
                            codes_seq = list(codes_list) if not isinstance(codes_list, list) else codes_list
                            names_seq = list(names_list) if not isinstance(names_list, list) else names_list
                            for ci, nm in zip(codes_seq, names_seq):
                                ci = int(ci)
                                if nm is not None and ci not in code_to_name:
                                    code_to_name[ci] = nm
                                    name_to_code[nm] = ci
                        except Exception:
                            continue

    # Build ordered labels based on discovered codes present in data
    present_codes = set()
    if name_to_code:
        # Use mapping to convert any observed labels to codes
        for nm in list(y_true) + list(y_pred):
            if nm in name_to_code:
                present_codes.add(name_to_code[nm])
    else:
        # Fallback: no mapping; use unique names as-is
        all_values = list(y_true) + list(y_pred)
        unique_values = set(val for val in all_values if val is not None)
        try:
            labels = sorted(list(unique_values))
        except TypeError:
            labels = sorted(list(unique_values), key=lambda x: str(x))

    if present_codes:
        for ci in sorted(present_codes):
            labels.append(code_to_name.get(ci, str(ci)))

    label_to_idx = {c: i for i, c in enumerate(labels)}

    # Accuracy
    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0

    # Confusion matrix
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    # Per-class precision/recall/F1 and macro averages
    per_class = []
    eps = 1e-12
    for i, c in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / max(eps, tp + fp)
        recall = tp / max(eps, tp + fn)
        f1 = 2 * precision * recall / max(eps, precision + recall)
        per_class.append({'class': c, 'precision': float(precision), 'recall': float(recall), 'f1': float(f1)})
    macro_precision = float(np.mean([x['precision'] for x in per_class])) if per_class else 0.0
    macro_recall = float(np.mean([x['recall'] for x in per_class])) if per_class else 0.0
    macro_f1 = float(np.mean([x['f1'] for x in per_class])) if per_class else 0.0

    # Top-k accuracy (if provided)
    topk_accuracy = None
    if topk_preds_col is not None and topk_preds_col in results_df.columns:
        topk_lists = results_df.select([topk_preds_col]).to_numpy().reshape(-1)
        hits = 0
        total = len(y_true)
        for truth, preds in zip(y_true, topk_lists):
            try:
                preds_list = list(preds) if not isinstance(preds, list) else preds
            except Exception:
                preds_list = []
            if truth in preds_list:
                hits += 1
        topk_accuracy = hits / max(1, total)

    # Print results if verbose
    if verbose:
        print("=== Classification Analysis Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        if topk_accuracy is not None:
            print(f"Top-k Accuracy: {topk_accuracy:.4f}")
        print(f"Number of Classes: {len(labels)}")
        print(f"Total Samples: {len(y_true)}")

    # Create plots if requested
    if create_plots:
        # Try to infer top_k from available columns
        inferred_top_k = None
        if topk_preds_col is not None and topk_preds_col in results_df.columns:
            try:
                # Parse the number from a name like f"{y_name}_Top{K}_Preds"
                m = re.search(r"_Top(\d+)_Preds$", topk_preds_col)
                if m:
                    inferred_top_k = int(m.group(1))
            except Exception:
                inferred_top_k = None
        _create_classification_plots(y_true, y_pred, cm, labels, per_class, y_true_col, top_k=inferred_top_k)

    return {
        'labels': labels,
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class': per_class,
        'confusion_matrix': cm,
        'topk_accuracy': topk_accuracy,
    }


def display_feature_importances_regression(
    saved_models_path,
    model_name,
    club_or_tournament=None,
    model_path=None,
    schema_path=None,
    top_n=50,
    bottom_n=50,
    return_df=True,
    verbose=True,
):
    """
    Display top/bottom features by absolute first-layer weight magnitude.
    Safely expands categorical embedding features to per-dimension names so
    the name list matches the model's first-layer input width.
    """

    model_path = resolve_model_path(saved_models_path, model_name)
    assert model_path.exists(), f"Model file not found: {model_path}"

    schema_path = resolve_schema_path(saved_models_path, model_name)
    assert schema_path.exists(), f"schema.json not found: {schema_path}"

    # Load schema
    schema = json.load(open(schema_path))
    numerical_feature_cols = schema.get('numerical_feature_cols', [])
    categorical_feature_cols = schema.get('categorical_feature_cols', [])
    cat_feature_info = schema.get('cat_feature_info', {})

    # Compute expected embedding output dims
    emb_dims = {col: (min(50, int(cat_feature_info.get(col, 0)) // 2) if int(cat_feature_info.get(col, 0)) > 0 else 0)
                for col in categorical_feature_cols}
    expected_input_dim = len(numerical_feature_cols) + sum(emb_dims.values())

    # Load state dict and find first 2D weight
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    weight_candidates = [(k, v) for k, v in state_dict.items() if torch.is_tensor(v) and v.ndim == 2]

    first_key, first_weight = None, None
    for preferred in ('layers.0.weight', 'model.0.weight'):
        w = state_dict.get(preferred)
        if torch.is_tensor(w) and w.ndim == 2:
            first_key, first_weight = preferred, w
            break
    if first_weight is None:
        # Prefer an exact expected_input_dim match, else the widest input
        exact = [(k, v) for k, v in weight_candidates if v.shape[1] == expected_input_dim]
        if exact:
            first_key, first_weight = exact[0]
        else:
            first_key, first_weight = max(weight_candidates, key=lambda kv: kv[1].shape[1])

    out_features, in_features = int(first_weight.shape[0]), int(first_weight.shape[1])

    if verbose:
        print("=== Feature Importance: First-Layer Weights ===")
        print(f"Using weight '{first_key}' with shape: {out_features} x {in_features}")
        if in_features != expected_input_dim:
            print(f"⚠️ Mismatch: schema effective input={expected_input_dim}, model expects={in_features}")

    # Build feature names expanded to embedding dims, then align to in_features
    expanded_embed_names = []
    for col in categorical_feature_cols:
        d = emb_dims.get(col, 0)
        if d > 0:
            expanded_embed_names.extend([f"_EMBEDDED_CAT_{col}_{i}" for i in range(d)])
    feature_names = numerical_feature_cols + expanded_embed_names

    if len(feature_names) < in_features:
        feature_names = feature_names + [f"UNKNOWN_MODEL_INPUT_{i}" for i in range(len(feature_names), in_features)]
    elif len(feature_names) > in_features:
        feature_names = feature_names[:in_features]

    # Compute importance (L1 across output neurons)
    W = first_weight.detach().cpu().numpy()
    importance = np.abs(W).sum(axis=0)

    # Final hard alignment guard
    if len(importance) != len(feature_names):
        if verbose:
            print(f"⚠️ Aligning lengths: importance={len(importance)} vs names={len(feature_names)}")
        n = min(len(importance), len(feature_names))
        importance = importance[:n]
        feature_names = feature_names[:n]

    imp_df = pl.DataFrame({
        'feature': feature_names,
        'importance': importance,
    })

    imp_df_sorted = imp_df.sort('importance', descending=True)

    if verbose:
        k_top = min(top_n, len(imp_df_sorted))
        k_bot = min(bottom_n, len(imp_df_sorted))
        print(f"\n✅ Created feature importance for {len(imp_df_sorted)} inputs")
        print(f"\nTop {k_top} features:")
        display(imp_df_sorted.head(k_top))
        print(f"\nBottom {k_bot} features:")
        display(imp_df_sorted.tail(k_bot).sort('importance'))

    return imp_df_sorted


def display_feature_importances(
    saved_models_path,
    model_name,
    **kwargs,
):
    """Wrapper: display feature importances for classification and regression.

    Auto-detects model type from the schema and routes accordingly. Always
    returns a Polars DataFrame. Classification without validation data falls
    back to first-layer weight visualization (same as regression).
    """
    top_n = kwargs.get('top_n', 50)
    bottom_n = kwargs.get('bottom_n', 50)
    verbose = kwargs.get('verbose', True)
    # Classification-only optional args
    val_df_or_tensors = kwargs.get('val_df_or_tensors')
    feature_cols = kwargs.get('feature_cols')

    try:
        schema_path = resolve_schema_path(saved_models_path, model_name)
        model_type = None
        if schema_path.exists():
            schema = json.load(open(schema_path))
            if schema.get('model_type') == 'classification' or ('class_to_idx' in schema):
                model_type = 'classification'
            else:
                model_type = 'regression'
        else:
            model_type = 'regression'
    except Exception:
        model_type = 'regression'

    if model_type == 'classification':
        # If validation data + explicit feature columns are provided, use permutation importance;
        # otherwise fall back to weight-based importance (reuses regression logic)
        if val_df_or_tensors is not None and feature_cols is not None:
            # Here, we expect a torch model; since we typically call the wrapper without a model
            # we only support the saved-model pathway unless caller passes a model explicitly.
            # So, route to permutation using saved model is not supported here; keep API strict.
            raise ValueError("display_feature_importances wrapper: permutation mode requires calling display_feature_importances_classification(model, feature_cols, val_df_or_tensors=...) directly")

        return display_feature_importances_classification(
            saved_models_path,
            model_name,
            val_df_or_tensors=None,
            top_n=top_n,
            bottom_n=bottom_n,
            return_df=True,
            verbose=verbose,
        )

    return display_feature_importances_regression(
        saved_models_path=saved_models_path,
        model_name=model_name,
        top_n=top_n,
        bottom_n=bottom_n,
        return_df=True,
        verbose=verbose,
    )

def display_feature_importances_classification(
    model_or_path: Any,
    feature_cols_or_model_name: Any,
    val_df_or_tensors: Optional[Any] = None,
    metric: str = 'accuracy',
    n_repeats: int = 1,
    device: Optional[str] = 'cuda',
    top_n: int = 50,
    bottom_n: int = 50,
    return_df: bool = False,
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Two modes:
    1) Permutation importance
       - model_or_path: torch.nn.Module
       - feature_cols_or_model_name: List[str] of feature column names
       - val_df_or_tensors: Polars DataFrame or (X_val, y_val) tensors
    2) First-layer weights from saved model
       - model_or_path: saved_models_path (pathlib.Path)
       - feature_cols_or_model_name: model_name (str)
       - val_df_or_tensors: must be None
    """
    # Mode 2: weights from saved model
    if val_df_or_tensors is None and not hasattr(model_or_path, 'parameters'):
        saved_models_path, model_name = model_or_path, feature_cols_or_model_name
        return display_feature_importances_regression(
            saved_models_path=saved_models_path,
            model_name=model_name,
            top_n=top_n,
            bottom_n=bottom_n,
            return_df=return_df,
            verbose=verbose,
        )

    # Mode 1: permutation importance
    assert metric in ('accuracy',), "Only 'accuracy' metric is supported for now"
    model: torch.nn.Module = model_or_path
    feature_cols: List[str] = feature_cols_or_model_name
    val_data: Any = val_df_or_tensors

    device_t = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model = model.to(device_t)
    model.eval()

    # Get validation tensors
    if isinstance(val_data, tuple):
        X_val, y_val = val_data
        assert isinstance(X_val, torch.Tensor) and isinstance(y_val, torch.Tensor)
    else:
        df = val_data
        X_val = torch.tensor(df.select(feature_cols).to_numpy(), dtype=torch.float32)
        y_val = torch.tensor(df.select([df.columns[-1]]).to_numpy().reshape(-1), dtype=torch.long)

    X_val = X_val.to(device_t)
    y_val = y_val.to(device_t)

    @torch.no_grad()
    def eval_accuracy(xb: torch.Tensor, yb: torch.Tensor) -> float:
        logits = model(xb)
        preds = torch.argmax(logits, dim=-1)
        return float((preds == yb).float().mean().item())

    baseline = eval_accuracy(X_val, y_val)

    rng = np.random.default_rng(123)
    importances = []
    X_val_cpu = X_val.detach().cpu().numpy()
    for j, col in enumerate(feature_cols):
        scores = []
        for _ in range(max(1, n_repeats)):
            X_perm = X_val_cpu.copy()
            rng.shuffle(X_perm[:, j])
            X_perm_t = torch.tensor(X_perm, dtype=torch.float32, device=device_t)
            score = eval_accuracy(X_perm_t, y_val)
            scores.append(baseline - score)
        importances.append({'feature': col, 'importance': float(np.mean(scores)), 'std': float(np.std(scores))})

    return pl.DataFrame(importances).sort('importance', descending=True)
