"""数据加载器：支持 .npz 图数据和 CSV 表格"""
import numpy as np
from scipy.sparse import csr_matrix
from typing import Dict


def load_classification_data(npz_path: str) -> Dict:
    """加载图节点分类数据。

    返回包含 adj, features, labels, train_idx, test_idx 等字段的字典。
    """
    data = np.load(npz_path)

    adj = csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=tuple(data["adj_shape"]),
    )
    features = csr_matrix(
        (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
        shape=tuple(data["attr_shape"]),
    )
    labels = data["labels"]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]

    train_labels = labels[train_idx]
    num_classes = len(np.unique(train_labels))
    features_dense = features.toarray().astype(np.float32)

    return {
        "adj": adj,
        "features": features_dense,
        "labels": labels.astype(np.int64),
        "train_idx": train_idx,
        "test_idx": test_idx,
        "num_classes": num_classes,
        "num_features": features.shape[1],
    }


def load_recommendation_data(data_dir: str) -> Dict:
    """Load sequence recommendation data from CSV directory

    Returns:
        dict with keys: train, test, users, items (all DataFrames)
    """
    import os
    import pandas as pd

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    user_df = pd.read_csv(os.path.join(data_dir, "user.csv"))
    item_df = pd.read_csv(os.path.join(data_dir, "item.csv"))

    return {
        "train": train_df,
        "test": test_df,
        "users": user_df,
        "items": item_df,
    }


def load_data(task_type: str, data_path: str) -> Dict:
    """Unified data loading entry point

    Args:
        task_type: "classification" or "recommendation"
        data_path: path to .npz file or data directory
    """
    if task_type == "classification":
        return load_classification_data(data_path)
    elif task_type == "recommendation":
        return load_recommendation_data(data_path)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
