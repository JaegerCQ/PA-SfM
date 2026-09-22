"""Time-interpolation adjoint with an optional collision-free vectorized path."""

import torch


def time_scatter_indices_unique(time_i0):
    """Check both interpolation destinations once; accepts CPU or CUDA indices."""
    indices = time_i0.detach().cpu().tolist()
    destinations = indices + [index + 1 for index in indices]
    return len(set(destinations)) == len(destinations)


def scatter_time_gradient(
    grad_out, time_i0, time_beta, n_bins, mode="legacy", indices_unique=None,
):
    """Scatter interpolation gradients while preserving colliding-write order.

    Callers may cache ``time_scatter_indices_unique(time_i0)`` when indices are
    constant. Vectorization is used only when all 2*T destinations are unique.
    Otherwise the original ordered accumulation is retained.
    """
    if mode not in {"legacy", "vectorized"}:
        raise ValueError(f"Unknown time scatter mode: {mode}")
    grad_out_c = grad_out.contiguous()
    result = torch.zeros(
        (grad_out_c.shape[0], n_bins),
        device=grad_out_c.device, dtype=torch.float32,
    )
    if mode == "vectorized":
        if indices_unique is None:
            indices_unique = time_scatter_indices_unique(time_i0)
        if indices_unique:
            # Match the original zero-plus-contribution operation, including
            # the sign of a zero-valued contribution.
            values0 = 0.0 + grad_out_c * (1.0 - time_beta)[None, :]
            values1 = 0.0 + grad_out_c * time_beta[None, :]
            result.index_copy_(1, time_i0, values0)
            result.index_copy_(1, time_i0 + 1, values1)
            return result

    for t in range(grad_out_c.shape[1]):
        idx0_t = time_i0[t]
        beta_t = time_beta[t]
        result[:, idx0_t] = result[:, idx0_t] + grad_out_c[:, t] * (1.0 - beta_t)
        result[:, idx0_t + 1] = result[:, idx0_t + 1] + grad_out_c[:, t] * beta_t
    return result
