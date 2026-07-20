from typing import Any, Optional

import torch

from rollingwam.utils.logging_config import get_logger

from .wam import WAM

logger = get_logger(__name__)


class RollingWAM(WAM):
    """Rolling-diffusion WAM: a sliding window of chunks denoised jointly under a
    diagonal noise ramp (front cleanest), conditioned on clean history.

    Sequence layout (video latents): [anchor frame | h context chunks | win window chunks],
    one chunk = `chunk_latents` latent frames. Actions pair per window chunk. Attention:
    anchor self-attends; context is block-causal, video-only; the window is bidirectional;
    action chunk i attends all video + its own chunk's actions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure_rolling()

    def configure_rolling(
        self,
        window_blocks: int = 4,
        num_context_chunks: int = 0,
        chunk_latents: int = 1,
        actions_per_chunk: int = 16,
        video_attends_actions: bool = True,
        context_routing: str = "split",
        num_inference_steps: int = 12,
        partial_window_prob: float = 0.0,
        partial_context_prob: float = 0.0,
    ):
        for a, b in (
            (self.train_video_scheduler, self.train_action_scheduler),
            (self.infer_video_scheduler, self.infer_action_scheduler),
        ):
            if a.shift != b.shift or a.num_train_timesteps != b.num_train_timesteps:
                raise ValueError(
                    "RollingWAM requires identical video/action schedulers (shift, "
                    f"num_train_timesteps); got video=({a.shift}, {a.num_train_timesteps}) "
                    f"vs action=({b.shift}, {b.num_train_timesteps})."
                )

        if window_blocks < 1:
            raise ValueError(f"`window_blocks` must be >= 1, got {window_blocks}")
        if num_context_chunks < 0:
            raise ValueError(f"`num_context_chunks` must be >= 0, got {num_context_chunks}")
        if chunk_latents < 1:
            raise ValueError(f"`chunk_latents` must be >= 1, got {chunk_latents}")
        if actions_per_chunk < 1:
            raise ValueError(f"`actions_per_chunk` must be >= 1, got {actions_per_chunk}")
        if num_inference_steps % window_blocks != 0:
            raise ValueError(
                f"`num_inference_steps` ({num_inference_steps}) must be divisible "
                f"by `window_blocks` ({window_blocks})"
            )
        if partial_window_prob + partial_context_prob > 1.0:
            raise ValueError(
                f"partial_window_prob + partial_context_prob must be <= 1, got "
                f"{partial_window_prob} + {partial_context_prob}"
            )
        if context_routing not in ("split", "shared"):
            raise ValueError(f"`context_routing` must be 'split' or 'shared', got {context_routing!r}")
        if num_context_chunks > 0 and partial_context_prob <= 0:
            logger.warning(
                "num_context_chunks=%d with partial_context_prob=0: inference will visit "
                "context sizes h=1..%d while the cache fills, but training never shows them. "
                "Set rolling.partial_context_prob > 0.",
                num_context_chunks, num_context_chunks - 1,
            )

        self.window_blocks = int(window_blocks)
        self.num_context_chunks = int(num_context_chunks)
        self.chunk_latents = int(chunk_latents)
        self.actions_per_chunk = int(actions_per_chunk)
        self.video_attends_actions = bool(video_attends_actions)
        self.context_routing = str(context_routing)
        self.num_inference_steps = int(num_inference_steps)
        self.partial_window_prob = float(partial_window_prob)
        self.partial_context_prob = float(partial_context_prob)
        self.rolling_reset()
        return self

    @classmethod
    def from_wan22_pretrained(cls, rolling: dict | None = None, **kwargs):
        video_dit_config = kwargs.get("video_dit_config", None)
        if not isinstance(video_dit_config, dict):
            raise ValueError("`video_dit_config` must be provided as dict for RollingWAM.")
        if bool(video_dit_config.get("action_conditioned", False)):
            raise ValueError("RollingWAM requires `video_dit_config['action_conditioned']=false`.")
        if video_dit_config.get("video_attention_mask_mode") != "bidirectional":
            raise ValueError(
                "RollingWAM requires `video_dit_config['video_attention_mask_mode']='bidirectional'` "
                "(the rolling mask is built by the model, not the video expert)."
            )
        model = super().from_wan22_pretrained(**kwargs)
        return model.configure_rolling(**(rolling or {}))

    ROLLING_KEYS = (
        "window_blocks", "num_context_chunks", "chunk_latents", "actions_per_chunk",
        "video_attends_actions", "context_routing", "num_inference_steps",
        "partial_window_prob", "partial_context_prob",
    )

    def get_rolling_config(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.ROLLING_KEYS}

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "rolling": self.get_rolling_config(),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = super().load_checkpoint(path, optimizer=optimizer)
        saved = payload.get("rolling")
        if saved is None:
            logger.warning(
                "Checkpoint has no rolling config (legacy/base checkpoint); keeping the "
                "current yaml rolling settings — verify they match how it was trained."
            )
            return payload

        expected_keys = set(self.ROLLING_KEYS)
        if not isinstance(saved, dict) or set(saved) != expected_keys:
            got = sorted(saved) if isinstance(saved, dict) else type(saved).__name__
            raise ValueError(
                "Checkpoint rolling config keys do not match: "
                f"expected={sorted(expected_keys)}, got={got}."
            )
        current = self.get_rolling_config()
        diff = {
            key: (current[key], saved[key])
            for key in self.ROLLING_KEYS
            if current[key] != saved[key]
        }
        if diff:
            logger.warning("Applying checkpoint rolling config over yaml: %s", diff)
        self.configure_rolling(**saved)
        return payload

    # ------------------------------------------------------------------ ladder

    def _rolling_ladder(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """S-rung schedule from the video inference scheduler.
        Returns (timesteps [S], deltas [S]); rung 0 = noisiest, rung S-1 = cleanest."""
        return self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=self.num_inference_steps,
            device=device,
            dtype=dtype,
        )

    def _window_rungs(self, win: int, phase: torch.Tensor) -> torch.Tensor:
        """Rung index per window block, front cleanest.
        phase: [B] in [0, sub). Returns [B, win]; block j gets (win-1-j)*sub + phase."""
        sub = self.num_inference_steps // self.window_blocks
        j = torch.arange(win, device=phase.device).view(1, win)
        return (win - 1 - j) * sub + phase.view(-1, 1)

    # ------------------------------------------------------------------ mask

    @torch.no_grad()
    def _build_rolling_attention_mask(
        self,
        ctx_chunks: int,
        win_chunks: int,
        tokens_per_frame: int,
        actions_per_chunk: int,
        device: torch.device,
        action_ctx_chunks: int = 0,
    ) -> torch.Tensor:
        """action_ctx_chunks > 0 = split routing: the action stream carries its own clean
        executed-action history, window actions read THAT instead of the video context
        (video history stays visible to window video only). 0 = shared routing: window
        actions read the video context directly."""
        nfpb = self.chunk_latents
        video_frames = 1 + (ctx_chunks + win_chunks) * nfpb
        video_seq_len = video_frames * tokens_per_frame
        action_seq_len = (action_ctx_chunks + win_chunks) * actions_per_chunk
        total = video_seq_len + action_seq_len

        frame_chunk = torch.full((video_frames,), -1, dtype=torch.long, device=device)
        frame_chunk[1:] = torch.arange((ctx_chunks + win_chunks) * nfpb, device=device) // nfpb
        token_chunk = frame_chunk.repeat_interleave(tokens_per_frame)
        is_window_token = token_chunk >= ctx_chunks

        mask = torch.zeros((total, total), dtype=torch.bool, device=device)

        v = slice(0, video_seq_len)
        mask[v, v] = token_chunk.view(-1, 1) >= token_chunk.view(1, -1)  # block-causal (anchor col included)
        mask[:video_seq_len, :video_seq_len][is_window_token] = True     # window rows: bidirectional over all video
        mask[:tokens_per_frame, :video_seq_len] = False
        mask[:tokens_per_frame, :tokens_per_frame] = True                # anchor self-attends only

        a_chunk = torch.arange(action_ctx_chunks + win_chunks, device=device).repeat_interleave(actions_per_chunk)
        a_is_hist = a_chunk < action_ctx_chunks
        a = slice(video_seq_len, total)

        av = torch.zeros((action_seq_len, video_seq_len), dtype=torch.bool, device=device)
        if action_ctx_chunks > 0:
            av[~a_is_hist] = (token_chunk < 0) | (token_chunk >= ctx_chunks)
        else:
            av[~a_is_hist] = True
        av[a_is_hist, :tokens_per_frame] = True
        mask[a, v] = av

        aa = torch.zeros((action_seq_len, action_seq_len), dtype=torch.bool, device=device)
        same_chunk = a_chunk.view(-1, 1) == a_chunk.view(1, -1)
        aa[~a_is_hist] = a_is_hist.view(1, -1) | same_chunk[~a_is_hist]
        aa[a_is_hist] = a_is_hist.view(1, -1) & (a_chunk.view(1, -1) <= a_chunk[a_is_hist].view(-1, 1))
        mask[a, a] = aa

        if self.video_attends_actions:
            win_a_chunk = a_chunk - action_ctx_chunks
            mask[v, a] = (
                token_chunk.view(-1, 1) == (ctx_chunks + win_a_chunk.view(1, -1))
            ) & (~a_is_hist).view(1, -1)
        return mask

    # ------------------------------------------------------------------ training

    def _draw_regime(self) -> tuple[int, int]:
        """(context chunks h, window chunks win) for this sample: steady / context-fill / grow."""
        H, W = self.num_context_chunks, self.window_blocks
        h, win = H, W
        p_w = self.partial_window_prob if W > 1 else 0.0
        p_c = self.partial_context_prob if H > 0 else 0.0
        r = torch.rand(()).item()
        if r < p_w:
            h, win = 0, int(torch.randint(1, W, ()).item())
        elif r < p_w + p_c:
            h = int(torch.randint(0, H, ()).item())
        return h, win

    def training_loss(self, sample, tiled: bool = False):
        H, W = self.num_context_chunks, self.window_blocks
        h, win = self._draw_regime() if self.dit.training else (H, W)

        if sample.get("proprio", None) is not None and h > 0:
            proprio = sample["proprio"]
            aspc = self.actions_per_chunk
            if proprio.shape[1] <= h * aspc:
                raise ValueError(
                    f"proprio has {proprio.shape[1]} steps; window start h*aspc={h * aspc} out of range"
                )
            sample = dict(sample)
            sample["proprio"] = proprio[:, h * aspc : h * aspc + 1]

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        batch_size, _, latent_t = input_latents.shape[:3]
        nfpb = self.chunk_latents
        total_chunks = (latent_t - 1) // nfpb
        if (latent_t - 1) % nfpb != 0 or total_chunks != H + W:
            raise ValueError(
                f"RollingWAM expects 1 + (W+H)*chunk_latents = {1 + (H + W) * nfpb} latent frames "
                f"(W={W}, H={H}, chunk_latents={nfpb}), got {latent_t}. Set data num_frames accordingly."
            )
        if action.shape[1] % total_chunks != 0:
            raise ValueError(
                f"Action horizon ({action.shape[1]}) must be divisible by W+H chunks ({total_chunks})."
            )
        aspc = action.shape[1] // total_chunks
        if aspc != self.actions_per_chunk:
            raise ValueError(
                f"Data supplies {aspc} actions per chunk but `actions_per_chunk={self.actions_per_chunk}`; "
                f"streaming inference relies on this being fixed."
            )

        latents = input_latents[:, :, : 1 + (h + win) * nfpb]
        window_action = action[:, h * aspc : (h + win) * aspc]
        window_action_is_pad = None if action_is_pad is None else action_is_pad[:, h * aspc : (h + win) * aspc]

        ha = h if self.context_routing == "split" else 0
        context_action = action[:, :ha * aspc] if ha > 0 else None

        device, dtype = latents.device, latents.dtype
        n_steps = float(self.train_video_scheduler.num_train_timesteps)
        ladder_t, _ = self._rolling_ladder(device, torch.float32)

        sub = self.num_inference_steps // W
        if self.dit.training and sub > 1:
            phase = torch.randint(0, sub, (batch_size,), device=device)
        else:
            phase = torch.zeros((batch_size,), dtype=torch.long, device=device)
        rungs = self._window_rungs(win, phase)                                   # [B, win]
        t_window_chunk = ladder_t[rungs]                                         # [B, win]

        timestep_video = torch.zeros((batch_size, latents.shape[2]), dtype=torch.float32, device=device)
        t_window_frame = t_window_chunk.repeat_interleave(nfpb, dim=1)           # [B, win*nfpb]
        timestep_video[:, 1 + h * nfpb :] = t_window_frame

        noise_video = torch.randn_like(latents)
        sigma_video = (timestep_video / n_steps).to(dtype).view(batch_size, 1, -1, 1, 1)
        noisy_latents = (1 - sigma_video) * latents + sigma_video * noise_video
        noisy_latents[:, :, : 1 + h * nfpb] = latents[:, :, : 1 + h * nfpb]
        target_video = noise_video - latents                                     # flow-matching velocity

        t_window_action = t_window_chunk.repeat_interleave(aspc, dim=1)          # [B, win*aspc]
        noise_action = torch.randn_like(window_action)
        sigma_action = (t_window_action / n_steps).to(dtype).unsqueeze(-1)
        noisy_window_action = (1 - sigma_action) * window_action + sigma_action * noise_action
        target_action = noise_action - window_action

        if ha > 0:
            action_tokens = torch.cat([context_action, noisy_window_action], dim=1)
            timestep_action = torch.cat(
                [torch.zeros((batch_size, ha * aspc), dtype=torch.float32, device=device), t_window_action],
                dim=1,
            )
        else:
            action_tokens = noisy_window_action
            timestep_action = t_window_action

        video_pre = self.video_expert.pre_dit(
            x=noisy_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_rolling_attention_mask(
            ctx_chunks=h,
            win_chunks=win,
            tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            actions_per_chunk=aspc,
            device=device,
            action_ctx_chunks=ha,
        )
        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        window_slice = slice(1 + h * nfpb, None)
        pred_w = pred_video[:, :, window_slice]
        target_w = target_video[:, :, window_slice]
        video_loss_frame = torch.nn.functional.mse_loss(
            pred_w.float(), target_w.float(), reduction="none"
        ).mean(dim=(1, 3, 4))                                                    # [B, win*nfpb]
        video_weight = self.train_video_scheduler.training_weight(
            t_window_frame.reshape(-1)
        ).reshape(batch_size, -1).to(video_loss_frame.device, dtype=video_loss_frame.dtype)
        if image_is_pad is not None:
            factor = int(self.vae.temporal_downsample_factor)
            latent_is_pad = image_is_pad[:, 1:].view(batch_size, -1, factor).all(dim=2)
            window_valid = (~latent_is_pad[:, h * nfpb : (h + win) * nfpb]).to(video_loss_frame.dtype)
        else:
            window_valid = torch.ones_like(video_loss_frame)
        video_weight = video_weight.clamp(min=0.05)
        valid_sum = window_valid.sum(dim=1).clamp(min=1.0)
        loss_video = ((video_loss_frame * video_weight * window_valid).sum(dim=1) / valid_sum).mean()

        pred_action_window = pred_action[:, ha * aspc :]
        action_loss_token = torch.nn.functional.mse_loss(
            pred_action_window.float(), target_action.float(), reduction="none"
        ).mean(dim=2)                                                            # [B, win*aspc]
        action_weight = self.train_action_scheduler.training_weight(
            t_window_action.reshape(-1)
        ).reshape(batch_size, -1).to(action_loss_token.device, dtype=action_loss_token.dtype)
        action_weight = action_weight.clamp(min=0.05)
        if window_action_is_pad is not None:
            action_valid = (~window_action_is_pad).to(action_loss_token.dtype)
        else:
            action_valid = torch.ones_like(action_loss_token)
        valid_sum = action_valid.sum(dim=1).clamp(min=1.0)
        loss_action = ((action_loss_token * action_weight * action_valid).sum(dim=1) / valid_sum).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    # ------------------------------------------------------------------ rolling inference

    def rolling_reset(self):
        """Clear streaming state (call at episode start)."""
        self._window_latents: Optional[torch.Tensor] = None   # [1, z, win*nfpb, h, w]
        self._window_action: Optional[torch.Tensor] = None    # [1, win*aspc, action_dim]
        self._rungs: list[int] = []
        self._raw_frames: Optional[torch.Tensor] = None       # [1, 3, N, H, W] in [-1, 1]
        self._executed_actions: list[torch.Tensor] = []        # emitted chunks, newest last
        self._push_count: int = 0
        self._cached_context: Optional[tuple] = None
        self._cached_negative: Optional[tuple] = None

    def _push_noise_chunk(self, z: int, latent_h: int, latent_w: int, aspc: int, seed: Optional[int]):
        """Append a pure-noise chunk (video + action) at the back of the window, rung 0.
        Deterministic per-push seed keeps CFG ranks and reruns identical."""
        gen = None
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(seed + self._push_count)
        self._push_count += 1
        v = torch.randn((1, z, self.chunk_latents, latent_h, latent_w), generator=gen).to(
            device=self.device, dtype=self.torch_dtype
        )
        a = torch.randn((1, aspc, self.action_expert.action_dim), generator=gen).to(
            device=self.device, dtype=self.torch_dtype
        )
        if self._window_latents is None or self._window_latents.shape[2] == 0:
            self._window_latents, self._window_action = v, a
        else:
            self._window_latents = torch.cat([self._window_latents, v], dim=2)
            self._window_action = torch.cat([self._window_action, a], dim=1)
        self._rungs.append(0)

    def _pop_front_chunk(self, aspc: int) -> tuple[torch.Tensor, torch.Tensor]:
        nfpb = self.chunk_latents
        video = self._window_latents[:, :, :nfpb]
        act = self._window_action[:, :aspc]
        self._window_latents = self._window_latents[:, :, nfpb:]
        self._window_action = self._window_action[:, aspc:]
        self._rungs = self._rungs[1:]
        self._executed_actions.append(act.detach().clone())
        self._executed_actions = self._executed_actions[-self.num_context_chunks :] if self.num_context_chunks > 0 else []
        return video, act

    @torch.no_grad()
    def _encode_context_latents(self) -> torch.Tensor:
        """Encode the visible clean history [anchor frame | h chunks] from the raw buffer.
        One-shot encode matches training's VAE statistics exactly."""
        frames = self._raw_frames.to(device=self.device, dtype=self.torch_dtype)
        return self._encode_video_latents(frames)

    @torch.no_grad()
    def _window_pass(
        self,
        ctx_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        ladder_t: torch.Tensor,
        ladder_delta: torch.Tensor,
        aspc: int,
        text_cfg_scale: float = 1.0,
        negative_context: Optional[torch.Tensor] = None,
        negative_context_mask: Optional[torch.Tensor] = None,
    ):
        """One denoise pass over the window against clean context; advances every rung by 1."""
        nfpb = self.chunk_latents
        win = len(self._rungs)
        h = (ctx_latents.shape[2] - 1) // nfpb

        if self.context_routing == "split" and self._executed_actions:
            ctx_actions = torch.cat(self._executed_actions, dim=1)
            ha = ctx_actions.shape[1] // aspc
        else:
            ctx_actions, ha = None, 0

        latents = torch.cat([ctx_latents, self._window_latents], dim=2)
        rungs = torch.tensor(self._rungs, device=self.device)
        t_chunk = ladder_t[rungs].to(torch.float32)                              # [win]
        timestep_video = torch.zeros((1, latents.shape[2]), dtype=torch.float32, device=self.device)
        timestep_video[0, ctx_latents.shape[2]:] = t_chunk.repeat_interleave(nfpb)
        t_window_action = t_chunk.repeat_interleave(aspc).unsqueeze(0)           # [1, win*aspc]
        if ha > 0:
            action_tokens = torch.cat([ctx_actions, self._window_action], dim=1)
            timestep_action = torch.cat(
                [torch.zeros((1, ha * aspc), dtype=torch.float32, device=self.device), t_window_action],
                dim=1,
            )
        else:
            action_tokens = self._window_action
            timestep_action = t_window_action

        def predict(ctx, ctx_mask):
            video_pre = self.video_expert.pre_dit(
                x=latents,
                timestep=timestep_video,
                context=ctx,
                context_mask=ctx_mask,
                fuse_vae_embedding_in_latents=bool(
                    getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
                ),
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=action_tokens,
                timestep=timestep_action,
                context=ctx,
                context_mask=ctx_mask,
            )
            attention_mask = self._build_rolling_attention_mask(
                ctx_chunks=h,
                win_chunks=win,
                tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                actions_per_chunk=aspc,
                device=self.device,
                action_ctx_chunks=ha,
            )
            tokens_out = self.mot(
                embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
                attention_mask=attention_mask,
                freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
                context_all={
                    "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                    "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
                },
                t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
            )
            return (
                self.video_expert.post_dit(tokens_out["video"], video_pre),
                self.action_expert.post_dit(tokens_out["action"], action_pre),
            )

        pred_video, pred_action = predict(context, context_mask)
        if text_cfg_scale != 1.0 and negative_context is not None:
            pred_video_neg, _ = predict(negative_context, negative_context_mask)
            pred_video = pred_video_neg + text_cfg_scale * (pred_video - pred_video_neg)
            # action keeps the conditional prediction

        pred_window = pred_video[:, :, ctx_latents.shape[2]:]
        pred_action_window = pred_action[:, ha * aspc :]
        for j, r in enumerate(self._rungs):
            delta = ladder_delta[r]
            vs = slice(j * nfpb, (j + 1) * nfpb)
            self._window_latents[:, :, vs] += pred_window[:, :, vs] * delta
            as_ = slice(j * aspc, (j + 1) * aspc)
            self._window_action[:, as_] += pred_action_window[:, as_] * delta
        self._rungs = [r + 1 for r in self._rungs]

    @torch.no_grad()
    def rolling_act(
        self,
        new_frames: torch.Tensor,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """One control step: feed newly observed frames, get the next action chunk.

        First call: `new_frames` [1, 3, 1, H, W] (single observation) — grows the window.
        Later calls: [1, 3, 4*chunk_latents, H, W] (the frames observed while executing
        the previous chunk). Returns {'action': [aspc, action_dim], 'video': predicted
        front chunk latents}.
        """
        self.eval()
        if new_frames.ndim != 5 or new_frames.shape[0] != 1 or new_frames.shape[1] != 3:
            raise ValueError(f"`new_frames` must be [1, 3, T, H, W], got {tuple(new_frames.shape)}")

        nfpb = self.chunk_latents
        H, W, S = self.num_context_chunks, self.window_blocks, self.num_inference_steps
        sub = S // W
        tdf = int(self.vae.temporal_downsample_factor)
        first_call = self._raw_frames is None

        if first_call:
            if new_frames.shape[2] != 1:
                raise ValueError(f"First call expects a single frame, got T={new_frames.shape[2]}")
            self._raw_frames = new_frames
        else:
            expected = tdf * nfpb
            if new_frames.shape[2] != expected:
                raise ValueError(
                    f"Expected {expected} new frames per call (one chunk), got {new_frames.shape[2]}"
                )
            self._raw_frames = torch.cat([self._raw_frames, new_frames], dim=2)
            max_frames = 1 + tdf * nfpb * H
            if self._raw_frames.shape[2] > max_frames:
                self._raw_frames = self._raw_frames[:, :, -max_frames:]

        if prompt is not None:
            if self._cached_context is None or self._cached_context[0] != prompt:
                self._cached_context = (prompt, self.encode_prompt(prompt))
            context, context_mask = self._cached_context[1]
        if context is None or context_mask is None:
            raise ValueError("Either `prompt` or `context`+`context_mask` must be provided.")
        context = context.to(device=self.device, dtype=self.torch_dtype)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        if proprio is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        negative_context = negative_context_mask = None
        if text_cfg_scale != 1.0:
            if self._cached_negative is None:
                self._cached_negative = self.encode_prompt(negative_prompt or "")
            negative_context, negative_context_mask = self._cached_negative
            if proprio is not None:
                negative_context, negative_context_mask = self._append_proprio_to_context(
                    context=negative_context,
                    context_mask=negative_context_mask,
                    proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
                )

        ctx_latents = self._encode_context_latents()
        z, latent_h, latent_w = ctx_latents.shape[1], ctx_latents.shape[3], ctx_latents.shape[4]
        aspc = self.actions_per_chunk
        ladder_t, ladder_delta = self._rolling_ladder(self.device, torch.float32)

        if first_call:
            for _ in range(W):
                self._push_noise_chunk(z, latent_h, latent_w, aspc, seed)
                for _ in range(sub):
                    self._window_pass(
                        ctx_latents, context, context_mask, ladder_t, ladder_delta, aspc,
                        text_cfg_scale, negative_context, negative_context_mask,
                    )
        else:
            self._push_noise_chunk(z, latent_h, latent_w, aspc, seed)
            for _ in range(sub):
                self._window_pass(
                    ctx_latents, context, context_mask, ladder_t, ladder_delta, aspc,
                    text_cfg_scale, negative_context, negative_context_mask,
                )

        video_front, action_front = self._pop_front_chunk(aspc)
        return {
            "action": action_front[0].detach().to(device="cpu", dtype=torch.float32),
            "video": video_front.detach(),
        }

    # ------------------------------------------------------------------ open-loop eval

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: Optional[int] = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Open-loop rollout for evaluation: rolling generation from one image, feeding the
        model's own emitted chunks back as context (no ground-truth camera available)."""
        del action, num_inference_steps, sigma_shift, rand_device, tiled
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)

        nfpb = self.chunk_latents
        latent_t = (num_video_frames - 1) // int(self.vae.temporal_downsample_factor) + 1
        if (latent_t - 1) % nfpb != 0:
            raise ValueError(f"num_video_frames gives {latent_t} latents, not 1 + k*chunk_latents")
        num_chunks = (latent_t - 1) // nfpb

        if prompt is not None:
            context, context_mask = self.encode_prompt(prompt)
        if context is not None and context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask is not None and context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)

        self.rolling_reset()
        emitted_video: list[torch.Tensor] = []
        emitted_action: list[torch.Tensor] = []
        anchor_latent = self._encode_input_image_latents_tensor(input_image=input_image)
        frames = input_image.unsqueeze(2)  # [1, 3, 1, H, W]
        for _ in range(num_chunks):
            out = self.rolling_act(
                new_frames=frames,
                context=context,
                context_mask=context_mask,
                proprio=proprio,
                negative_prompt=negative_prompt,
                text_cfg_scale=text_cfg_scale,
                seed=seed,
            )
            emitted_action.append(out["action"])
            emitted_video.append(out["video"])
            # VAE maps 1+k latents <-> 1+4k frames; an isolated chunk decodes wrong
            stream = torch.cat([anchor_latent] + emitted_video, dim=2)
            decoded = self._decode_latents_tensor(stream)
            frames = decoded[:, :, -int(self.vae.temporal_downsample_factor) * out["video"].shape[2] :]
        self.rolling_reset()

        video_latents = torch.cat([anchor_latent] + emitted_video, dim=2)
        video = self._decode_latents(video_latents)
        return {
            "video": video,
            "action": torch.cat(emitted_action, dim=0)[:action_horizon],
        }

    @torch.no_grad()
    def _decode_latents_tensor(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(dtype=self.torch_dtype), device=self.device)
        return frames.clamp(-1, 1)

