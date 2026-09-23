"""Experimental capture of a complete, fixed 600-step localization schedule.

Every replay starts a fresh independent Adam optimization. All 600 original
CPU bias-correction values are recorded separately, so replay never advances
an optimizer between independent sensor batches. This deliberately preserves
the non-capturable, single-tensor Adam arithmetic rather than changing to the
different capturable=True floating-point path. Do not use it as a generic
single-step optimizer graph.
"""

import time

import torch
from torch.optim import _functional


@torch.no_grad()
def original_functional_adam(parameter, gradient, exp_avg, exp_avg_sq, cpu_step, lr):
    """The same functional call made by Adam(foreach=False), without its wrapper."""
    _functional.adam(
        [parameter], [gradient], [exp_avg], [exp_avg_sq], [], [cpu_step],
        foreach=False, capturable=False, differentiable=False, fused=False,
        grad_scale=None, found_inf=None, has_complex=False,
        amsgrad=False, beta1=0.9, beta2=0.999, lr=lr, weight_decay=0.0,
        eps=1e-8, maximize=False)


class FineSearchGraph:
    def __init__(self, localization, xyz_src, pc, target, initial,
                 total_epochs=600, lr=0.0005, sigma_start=1.5e-3,
                 sigma_target=0.1e-3, t_start=500, n_time=500, delta_t=100e-9):
        if not initial.is_cuda or total_epochs != 600:
            raise ValueError("This experiment requires CUDA and the full 600-step schedule")
        self.localization, self.xyz_src, self.pc = localization, xyz_src, pc
        self.target = target.detach().clone()
        self.initial = initial.detach().clone()
        self.parameter = torch.nn.Parameter(initial.detach().clone())
        self.exp_avg = torch.zeros_like(self.parameter)
        self.exp_avg_sq = torch.zeros_like(self.parameter)
        self.cpu_step = torch.tensor(0.0, device="cpu")
        self.total_epochs, self.lr = total_epochs, lr
        self.sigma_start, self.sigma_target = sigma_start, sigma_target
        self.t_start, self.n_time, self.delta_t = t_start, n_time, delta_t
        self.setup_seconds = 0.0
        self.capture_seconds = 0.0

        started = time.perf_counter()
        warmup_stream = torch.cuda.Stream(device=initial.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(initial.device))
        with torch.cuda.stream(warmup_stream):
            self.cpu_step.zero_()
            self._reset_cuda_state()
            self._run_schedule()
        torch.cuda.current_stream(initial.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(initial.device)

        # The CPU counter executes while capturing and embeds steps 1..600.
        # Replays reset all GPU state; no CPU counter is needed during replay.
        self.cpu_step.zero_()
        self.graph = torch.cuda.CUDAGraph()
        capture_started = time.perf_counter()
        with torch.cuda.graph(self.graph):
            self._reset_cuda_state()
            self.losses = self._run_schedule()
        self.capture_seconds = time.perf_counter() - capture_started
        self.setup_seconds = time.perf_counter() - started
        if self.cpu_step.item() != total_epochs:
            raise RuntimeError("Capture did not record all original Adam step indices")

    @torch.no_grad()
    def _reset_cuda_state(self):
        self.parameter.copy_(self.initial)
        self.exp_avg.zero_()
        self.exp_avg_sq.zero_()

    def _run_schedule(self):
        for epoch in range(self.total_epochs):
            self.parameter.grad = None
            sigma = self.sigma_start * (self.sigma_target / self.sigma_start) ** min(
                1.0, epoch / (self.total_epochs * 0.8))
            prediction = self.localization.FORWARD_BATCH(
                self.parameter, self.xyz_src, self.pc, self.t_start, self.n_time,
                self.delta_t, vs=self.localization.VS_DEFAULT, sigma=sigma)
            losses = self.localization.neg_corr_loss_batch(prediction, self.target)
            losses.sum().backward()
            original_functional_adam(self.parameter, self.parameter.grad,
                                     self.exp_avg, self.exp_avg_sq, self.cpu_step, self.lr)
        return losses.detach()

    def __call__(self, target, initial):
        if target.shape != self.target.shape or initial.shape != self.initial.shape:
            raise ValueError("A captured schedule requires the original fixed batch shapes")
        with torch.no_grad():
            self.target.copy_(target)
            self.initial.copy_(initial)
        self.graph.replay()
        # Preserve legacy selection: final pre-update losses, post-update position.
        return self.parameter.detach(), self.losses
