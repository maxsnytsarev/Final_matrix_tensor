from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import CompletionResult, TensorCompletionBaseline, all_metrics

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


class _SpatialAttentionModule(nn.Module):
    def __init__(self, n_feats: int):
        super().__init__()
        self.att1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.att2(self.relu(self.att1(x))))


class _RDBConv(nn.Module):
    def __init__(self, in_channels: int, grow_rate: int, ksize: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, grow_rate, ksize, padding=(ksize - 1) // 2, stride=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return torch.cat((x, out), dim=1)


class _RDB(nn.Module):
    def __init__(self, tensor_num_channels: int, grow_rate0: int, grow_rate: int, n_conv_layers: int):
        super().__init__()
        g0 = grow_rate0
        g = grow_rate
        c = n_conv_layers

        convs = []
        for i in range(c):
            convs.append(_RDBConv(g0 + i * g, g))
        self.convs = nn.Sequential(*convs)

        self.LFF = nn.Conv2d(g0 + c * g, g0, 1, padding=0, stride=1)
        self.in_conv = nn.Conv2d(in_channels=tensor_num_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=tensor_num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        x = self.LFF(self.convs(x)) + x
        x = self.out_conv(x)
        return x


class _RPCABlock(nn.Module):
    def __init__(self, tensor_num_channels: int):
        super().__init__()

        self.lamb = nn.Parameter(torch.ones(1) * 0.001, requires_grad=True)
        self.delta = nn.Parameter(torch.ones(1) * 0.001, requires_grad=True)
        self.mu = nn.Parameter(torch.ones(1) * 0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1) * 0.001, requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1) * 0.001, requires_grad=True)

        self.Proximal_P = _RDB(tensor_num_channels, grow_rate0=64, grow_rate=32, n_conv_layers=8)
        self.Proximal_Q = _RDB(tensor_num_channels, grow_rate0=64, grow_rate=32, n_conv_layers=6)

    def tensor_product(self, l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        lf = torch.fft.fft(torch.squeeze(l), n=l.shape[-1], dim=2).permute(2, 0, 1)
        rf = torch.fft.fft(torch.squeeze(r), n=r.shape[-1], dim=2).permute(2, 0, 1)
        gf = torch.matmul(lf, rf).permute(1, 2, 0)
        return torch.unsqueeze(torch.fft.irfft(gf, n=r.shape[-1], dim=2), 0)

    def decom_solution(self, l_k: torch.Tensor, r_k: torch.Tensor, c_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = torch.fft.fft(torch.squeeze(c_k), n=c_k.shape[-1], dim=2).permute(2, 0, 1)
        l = torch.fft.fft(torch.squeeze(l_k), n=l_k.shape[-1], dim=2).permute(2, 0, 1)
        r = torch.fft.fft(torch.squeeze(r_k), n=r_k.shape[-1], dim=2).permute(2, 0, 1)

        li = torch.matmul(
            torch.matmul(c, torch.transpose(torch.conj(r), 1, 2)),
            torch.linalg.pinv(torch.matmul(r, torch.transpose(torch.conj(r), 1, 2)), rcond=1e-4),
        ).permute(1, 2, 0)

        ri = torch.matmul(
            torch.matmul(
                torch.linalg.pinv(torch.matmul(torch.transpose(torch.conj(l), 1, 2), l), rcond=1e-4),
                torch.transpose(torch.conj(l), 1, 2),
            ),
            c,
        ).permute(1, 2, 0)

        return (
            torch.unsqueeze(torch.fft.irfft(li, n=l_k.shape[-1], dim=2), 0),
            torch.unsqueeze(torch.fft.irfft(ri, n=r_k.shape[-1], dim=2), 0),
        )

    def forward(
        self,
        l: torch.Tensor,
        r: torch.Tensor,
        c: torch.Tensor,
        e: torch.Tensor,
        t: torch.Tensor,
        p: torch.Tensor,
        q: torch.Tensor,
        l1: torch.Tensor,
        l2: torch.Tensor,
        l3: torch.Tensor,
        omega: torch.Tensor,
        w: torch.Tensor,
        omega_c: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        device = omega.device

        psi_c = self.mu + self.alpha
        psi_c_term = l1 - l2 + self.mu * omega - self.mu * e - self.mu * t + self.alpha * p
        c_k = torch.div(torch.mul(w, self.tensor_product(l, r)) + psi_c_term, w + psi_c)

        l_k, r_k = self.decom_solution(l, r, c_k)

        psi_e = self.mu + self.beta
        psi_e_term = (l1 - l3 + self.mu * omega - self.mu * c_k - self.mu * t + self.beta * q) / psi_e
        e_k = torch.mul(torch.sign(psi_e_term), nn.functional.relu(torch.abs(psi_e_term) - self.lamb / psi_e))

        y = omega - c_k - e_k + l1 / self.mu
        t_k = torch.mul(y, omega_c) + torch.mul(y, omega) * torch.min(
            torch.tensor(1.0, device=device),
            self.delta / (torch.norm(torch.mul(y, omega), p='fro') + 1e-6),
        )

        p_k = self.Proximal_P(c_k + l2 / (self.alpha + 1e-6))
        q_k = self.Proximal_Q(e_k + l3 / (self.beta + 1e-6))

        l1_k = l1 + self.mu * (omega - c_k - e_k - t_k)
        l2_k = l2 + self.alpha * (c_k - p_k)
        l3_k = l3 + self.beta * (e_k - q_k)

        return l_k, r_k, c_k, e_k, t_k, p_k, q_k, l1_k, l2_k, l3_k


class _RPCANet(nn.Module):
    def __init__(self, n_iter: int, tensor_num_channels: int):
        super().__init__()
        self.n_iter = n_iter
        self.tensor_num_channels = tensor_num_channels
        self.att_module = _SpatialAttentionModule(self.tensor_num_channels)

        blocks = []
        for _ in range(self.n_iter):
            blocks.append(_RPCABlock(self.tensor_num_channels))
        self.network = nn.ModuleList(blocks)

    def forward(self, data: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        device = data.device

        omega_d = data
        omega_c = torch.ones(1, device=device, dtype=omega.dtype) - omega

        w = self.att_module(omega_d)

        omega_d = torch.mul(omega_d, omega)
        c = omega_d

        l1 = torch.zeros(c.size(), device=device)
        l2 = torch.zeros(c.size(), device=device)
        l3 = torch.zeros(c.size(), device=device)
        e = torch.zeros(c.size(), device=device)
        t = torch.zeros(c.size(), device=device)
        p = torch.zeros(c.size(), device=device)
        q = torch.zeros(c.size(), device=device)

        l = torch.ones((self.tensor_num_channels, 10, omega_d.shape[-1]), device=device, dtype=data.dtype) / 1e2
        r = torch.ones((10, omega_d.shape[-2], omega_d.shape[-1]), device=device, dtype=data.dtype) / 1e2

        for i in range(self.n_iter):
            l, r, c, e, t, p, q, l1, l2, l3 = self.network[i](
                l,
                r,
                c,
                e,
                t,
                p,
                q,
                l1,
                l2,
                l3,
                omega_d,
                w,
                omega_c,
            )
        return c


def _grid_points(size: int, patch_size: int) -> list[int]:
    patch_size = max(1, int(patch_size))
    if size <= patch_size:
        return [0]
    pts = list(range(0, size - patch_size + 1, patch_size))
    if pts[-1] != size - patch_size:
        pts.append(size - patch_size)
    return pts


def _to_hwb(arr: np.ndarray, expected_channels: int) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got shape {x.shape}")
    if x.shape[-1] == expected_channels:
        return x
    if x.shape[0] == expected_channels:
        return np.transpose(x, (2, 1, 0))
    raise ValueError(
        f"Could not infer HxWxC layout for expected_channels={expected_channels}. "
        f"Got shape={x.shape}."
    )


@dataclass
class AGTCHSIPretrained(TensorCompletionBaseline):
    checkpoint_path: str | Path
    tensor_num_channels: int
    n_iter: int = 10
    patch_size: int = 64
    device: str = "auto"  # auto | cpu | cuda
    show_tqdm: bool = True
    verbose: bool = False

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _load_model(self, device: torch.device) -> _RPCANet:
        model = _RPCANet(n_iter=int(self.n_iter), tensor_num_channels=int(self.tensor_num_channels)).to(device)
        checkpoint = torch.load(str(self.checkpoint_path), map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def _infer(self, model: _RPCANet, data_hwb: np.ndarray, mask_hwb: np.ndarray) -> np.ndarray:
        data_hwb = _to_hwb(data_hwb, expected_channels=self.tensor_num_channels).astype(np.float32)
        mask_hwb = _to_hwb(mask_hwb, expected_channels=self.tensor_num_channels).astype(np.float32)

        data_bwh = np.transpose(data_hwb, (2, 1, 0))
        mask_bwh = np.transpose(mask_hwb, (2, 1, 0))

        _, width, height = data_bwh.shape
        w_grid = _grid_points(width, self.patch_size)
        h_grid = _grid_points(height, self.patch_size)

        out_hwb = np.zeros((height, width, self.tensor_num_channels), dtype=np.float32)

        rows = tqdm(h_grid, desc="[AGTC] rows", leave=False, disable=not self.show_tqdm)
        device = next(model.parameters()).device

        for h in rows:
            cols = tqdm(w_grid, desc="[AGTC] cols", leave=False, disable=not self.show_tqdm)
            for w in cols:
                data_patch = data_bwh[:, w:w + self.patch_size, h:h + self.patch_size]
                mask_patch = mask_bwh[:, w:w + self.patch_size, h:h + self.patch_size]

                x = torch.from_numpy(data_patch).unsqueeze(0).to(device)
                omega = torch.from_numpy(mask_patch).unsqueeze(0).to(device)

                with torch.no_grad():
                    c = model(x, omega)

                c_np = torch.squeeze(c).detach().cpu().numpy()
                patch_hwb = np.transpose(c_np, (2, 1, 0))
                ph, pw, _ = patch_hwb.shape
                out_hwb[h:h + ph, w:w + pw, :] = patch_hwb

        return out_hwb

    def fit_transform(
        self,
        observed_tensor: np.ndarray,
        mask: np.ndarray,
        full_tensor: np.ndarray | None = None,
    ) -> CompletionResult:
        if observed_tensor.ndim != 3:
            raise ValueError(f"AGTCHSIPretrained expects 3D tensor, got shape {observed_tensor.shape}")

        checkpoint = Path(self.checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        observed_hwb = _to_hwb(np.asarray(observed_tensor, dtype=float), expected_channels=self.tensor_num_channels)
        mask_hwb = _to_hwb(np.asarray(mask, dtype=bool), expected_channels=self.tensor_num_channels)

        device = self._resolve_device()
        if self.verbose:
            print(f"[AGTC] loading model on device={device} from checkpoint={checkpoint}")

        model = self._load_model(device=device)
        recovered = self._infer(model=model, data_hwb=observed_hwb, mask_hwb=mask_hwb)
        recovered = np.clip(recovered.astype(float), 0.0, 1.0)

        train_metrics = test_metrics = None
        if full_tensor is not None:
            full_hwb = _to_hwb(np.asarray(full_tensor, dtype=float), expected_channels=self.tensor_num_channels)
            train_metrics = all_metrics(full_hwb, recovered, mask_hwb)
            test_metrics = all_metrics(full_hwb, recovered, ~mask_hwb)

        return CompletionResult(
            tensor=recovered,
            factors=None,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            info={
                "checkpoint_path": str(checkpoint),
                "n_iter": int(self.n_iter),
                "patch_size": int(self.patch_size),
                "device": str(device),
            },
        )
