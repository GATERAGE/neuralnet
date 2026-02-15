#!/usr/bin/env python3
# simplemind_jax.py
"""
SimpleMind (JAX) — lightweight MLP classifier for RAGE policy/reranking.

Purpose in RAGE
- Rerank retrieved chunks (capsules) using engineered features (vector sim, BM25 score, integrity tier, freshness, etc.)
- Trust gating (e.g., "is this capsule reliable enough for default retrieval?")
- Routing decisions (local vs remote model, promote vs keep in cache)

Notes
- This is a compact, JIT-friendly JAX/Optax implementation.
- For universal access inside neuralnet (PyTorch stack), consider `simplemind_torch.py`
  to avoid adding a second ML framework dependency.

License: follow repo license.
"""

from __future__ import annotations

import os
import logging
import numpy as np
import h5py
from multiprocessing import Process, Queue, cpu_count
from queue import Empty
from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from jax import jit


class HDF5DataLoader:
    """
    Multiprocess HDF5 dataloader.
    Assumes HDF5 file has datasets:
      - X: (N, D) float/bytes
      - y: (N,) or (N,1) labels

    This is optional. RAGE often builds feature matrices in-memory.
    """
    def __init__(self, h5_path: str, batch_size: int, shuffle: bool = True, num_workers: Optional[int] = None):
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = max(1, num_workers or cpu_count() // 2)

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        with h5py.File(self.h5_path, "r") as hf:
            self.num_samples = hf["X"].shape[0]

        self.processes = []
        self.data_queue: Queue = Queue(maxsize=self.num_workers * 4)

    def _worker_loop(self, indices_queue: Queue):
        with h5py.File(self.h5_path, "r") as hf:
            X_dset, y_dset = hf["X"], hf["y"]
            while True:
                try:
                    indices = indices_queue.get(timeout=1)
                    if indices is None:
                        break
                    indices = np.sort(indices)
                    self.data_queue.put((X_dset[indices], y_dset[indices]))
                except Empty:
                    continue

    def __iter__(self):
        indices_queue: Queue = Queue()

        self.processes = [Process(target=self._worker_loop, args=(indices_queue,)) for _ in range(self.num_workers)]
        for p in self.processes:
            p.daemon = True
            p.start()

        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, self.num_samples, self.batch_size):
            indices_queue.put(indices[i:i + self.batch_size])

        for _ in range(self.num_workers):
            indices_queue.put(None)

        return self

    def __next__(self):
        try:
            return self.data_queue.get(timeout=30)
        except Empty:
            self.shutdown()
            raise StopIteration

    def __len__(self) -> int:
        return int(np.ceil(self.num_samples / self.batch_size))

    def shutdown(self):
        for p in self.processes:
            if p.is_alive():
                p.terminate()
            p.join()


class SimpleMind:
    """
    SimpleMind: a compact MLP binary classifier (logits -> sigmoid).

    Typical RAGE use:
      - X = engineered features per candidate capsule (shape B,D)
      - y = label: 1 if useful/verified/relevant, else 0
    """
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int = 1,
        activation: str = "relu",
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        lr_schedule_opts: Optional[Dict] = None,
        seed: int = 0,
    ):
        self.input_size = int(input_size)
        self.hidden_sizes = list(map(int, hidden_sizes))
        self.output_size = int(output_size)
        self.learning_rate = float(learning_rate)
        self.optimizer_name = optimizer
        self.activation_name = activation
        self.lr_schedule_opts = lr_schedule_opts or {}
        self.rng = random.PRNGKey(seed)
        self._setup_logging()

        if self.input_size > 0:
            self.re_initialize(self.input_size)
        else:
            self.params = {}
            self.opt_state = None

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.log = logging.getLogger("SimpleMind(JAX)")

    def _initialize_parameters(self):
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        params = {}
        key = self.rng

        for i in range(len(layer_sizes) - 1):
            key, layer_rng = random.split(key)
            # He-ish scaling (works for relu-ish)
            stddev = jnp.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            params[f"W{i}"] = random.normal(layer_rng, (layer_sizes[i], layer_sizes[i + 1])) * stddev
            params[f"b{i}"] = jnp.zeros((layer_sizes[i + 1],), dtype=jnp.float32)

        self.rng = key
        return params

    def re_initialize(self, input_size: int):
        self.input_size = int(input_size)
        self.params = self._initialize_parameters()
        self.optimizer = self._get_optimizer()
        self.opt_state = self.optimizer.init(self.params)

        self._update_step = jit(self._make_update_step())
        self._jit_compute_loss_and_metrics = jit(self._compute_loss_and_metrics)

    def _get_optimizer(self):
        if self.lr_schedule_opts:
            schedule = optax.exponential_decay(
                init_value=self.learning_rate,
                transition_steps=int(self.lr_schedule_opts.get("steps", 1000)),
                decay_rate=float(self.lr_schedule_opts.get("decay", 0.9)),
            )
        else:
            schedule = self.learning_rate

        optimizers = {
            "adam": optax.adam(schedule),
            "rmsprop": optax.rmsprop(schedule),
            "sgd": optax.sgd(schedule),
        }
        return optimizers.get(self.optimizer_name, optax.adam(schedule))

    def _activation(self):
        activations = {
            "relu": jax.nn.relu,
            "leaky_relu": jax.nn.leaky_relu,
            "gelu": jax.nn.gelu,
            "sigmoid": jax.nn.sigmoid,
            "tanh": jnp.tanh,
        }
        return activations.get(self.activation_name, jax.nn.relu)

    def forward(self, params, X: jnp.ndarray) -> jnp.ndarray:
        act = self._activation()
        out = X
        for i in range(len(self.hidden_sizes)):
            out = act(jnp.dot(out, params[f"W{i}"]) + params[f"b{i}"])
        out = jnp.dot(out, params[f"W{len(self.hidden_sizes)}"]) + params[f"b{len(self.hidden_sizes)}"]
        return out

    def _loss_fn(self, params, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        logits = self.forward(params, X)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y))

    def _compute_loss_and_metrics(self, params, X: jnp.ndarray, y: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        loss = self._loss_fn(params, X, y)
        logits = self.forward(params, X)

        y_pred = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.int32)
        y_true = y.astype(jnp.int32).reshape(-1, 1)

        accuracy = jnp.mean(y_pred == y_true)
        tp = jnp.sum((y_pred == 1) & (y_true == 1))
        fp = jnp.sum((y_pred == 1) & (y_true == 0))
        fn = jnp.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

        return {"loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

    def _make_update_step(self):
        def update_step(params, opt_state, X_batch, y_batch):
            loss, grads = jax.value_and_grad(self._loss_fn)(params, X_batch, y_batch)
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss
        return update_step

    def run_train_step(self, X_batch: jnp.ndarray, y_batch: jnp.ndarray) -> jnp.ndarray:
        self.params, self.opt_state, loss = self._update_step(self.params, self.opt_state, X_batch, y_batch)
        return loss

    def run_evaluation(self, data_loader: HDF5DataLoader) -> Dict[str, float]:
        total = {}
        n = 0
        try:
            for X_np, y_np in data_loader:
                X = jnp.array(X_np)
                y = jnp.array(y_np)
                metrics = self._jit_compute_loss_and_metrics(self.params, X, y)
                for k, v in metrics.items():
                    total[k] = total.get(k, 0.0) + float(v)
                n += 1
        finally:
            data_loader.shutdown()
        return {k: v / max(n, 1) for k, v in total.items()} if n > 0 else {}

    def predict_proba(self, X: jnp.ndarray) -> jnp.ndarray:
        logits = self.forward(self.params, X)
        return jax.nn.sigmoid(logits)
