"""GPU autoencoder anomaly detector with a scikit-learn-style API.

Imported by run_sample_size_experiment.py and notebook 02 as a 6th unsupervised
model. Anomaly score = per-row reconstruction error (MSE across features).
decision_function returns the NEGATED error so the project's
anomaly_scores(model, X, uses_fit_predict=False) == -decision_function(X)
recovers "higher = more anomalous", matching y=1 (fraud). anomaly_utils.py is
left untouched as the metric single source of truth.
"""
import numpy as np
import torch
import torch.nn as nn


class _AE(nn.Module):
    """Symmetric autoencoder: input -> encoder_dims (bottleneck) -> input.

    ReLU on hidden layers, linear output (inputs are standardized, can be
    negative, so no bounded output activation).
    """

    def __init__(self, input_dim, encoder_dims):
        super().__init__()
        dims = [input_dim, *encoder_dims]
        enc = []
        for i in range(len(dims) - 1):
            enc += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)
        dec = []
        rev = list(reversed(dims))
        for i in range(len(rev) - 1):
            dec.append(nn.Linear(rev[i], rev[i + 1]))
            if i < len(rev) - 2:          # ReLU on hidden, linear on the output
                dec.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderDetector:
    """Reconstruction-error anomaly detector trained on the unlabeled features."""

    def __init__(self, encoder_dims=(20, 14), epochs=30, batch_size=2048,
                 lr=1e-3, device=None, random_state=42, verbose=True):
        self.encoder_dims = tuple(encoder_dims)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = None

    def _to_tensor(self, X):
        return torch.from_numpy(np.asarray(X, dtype=np.float32))

    def fit(self, X):
        torch.manual_seed(self.random_state)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.random_state)
        data = self._to_tensor(X).to(self.device)
        n, input_dim = data.shape
        self.model_ = _AE(input_dim, self.encoder_dims).to(self.device)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        g = torch.Generator().manual_seed(self.random_state)  # CPU generator, deterministic
        self.model_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g).to(self.device)
            for i in range(0, n, self.batch_size):
                batch = data[perm[i:i + self.batch_size]]
                opt.zero_grad()
                loss = loss_fn(self.model_(batch), batch)
                loss.backward()
                opt.step()
        if self.device == "cuda":
            torch.cuda.synchronize()  # CUDA is async; finish before the caller stops its timer
        if self.verbose:
            print(f"[AutoencoderDetector] trained on {self.device}", flush=True)
        return self

    def decision_function(self, X):
        if self.model_ is None:
            raise RuntimeError("Call fit() before decision_function().")
        self.model_.eval()
        data = self._to_tensor(X).to(self.device)
        errs = torch.zeros(data.shape[0], device=self.device)
        with torch.no_grad():
            for i in range(0, data.shape[0], self.batch_size):
                batch = data[i:i + self.batch_size]
                errs[i:i + batch.shape[0]] = ((self.model_(batch) - batch) ** 2).mean(dim=1)
        if self.device == "cuda":
            torch.cuda.synchronize()
        return (-errs).cpu().numpy()   # NEGATED: higher decision_function = more normal
