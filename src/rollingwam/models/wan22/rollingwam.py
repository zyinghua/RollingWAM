from typing import Any, Optional

import torch

from rollingwam.utils.logging_config import get_logger

from .wam import WAM

logger = get_logger(__name__)


class RollingWAM(WAM):
    """Rolling-diffusion WAM: a sliding window of chunks denoised jointly under a
    diagonal noise ramp (front cleanest), conditioned on clean video history.

    Sequence layout (video latents): [anchor frame | h context chunks | win window chunks],
    one chunk = `chunk_latents` latent frames. Actions pair per window chunk. Attention:
    anchor self-attends; context is block-causal; window video is bidirectional over video;
    action chunk i attends all video + its own chunk's actions; video never attends actions.
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
        init_schedule_prob: float = 0.0,
        partial_context_prob: float = 0.0,
        obs_offset_prob: float = 0.0,
        obs_offset_range: int = 0,
    ):
        for a, b in (
            (self.train_video_scheduler, self.train_action_scheduler),
            (self.infer_video_scheduler, self.infer_action_scheduler),
            (self.train_video_scheduler, self.infer_video_scheduler),
        ):
            if a.shift != b.shift or a.num_train_timesteps != b.num_train_timesteps:
                raise ValueError(
                    "RollingWAM requires identical schedulers across video/action and "
                    f"train/infer (shift, num_train_timesteps); got ({a.shift}, "
                    f"{a.num_train_timesteps}) vs ({b.shift}, {b.num_train_timesteps})."
                )

        if window_blocks < 1:
            raise ValueError(f"`window_blocks` must be >= 1, got {window_blocks}")
        if num_context_chunks < 0:
            raise ValueError(f"`num_context_chunks` must be >= 0, got {num_context_chunks}")
        if chunk_latents < 1:
            raise ValueError(f"`chunk_latents` must be >= 1, got {chunk_latents}")
        if actions_per_chunk < 1:
            raise ValueError(f"`actions_per_chunk` must be >= 1, got {actions_per_chunk}")
        for name, value in (
            ("init_schedule_prob", init_schedule_prob),
            ("partial_context_prob", partial_context_prob),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"`{name}` must be in [0, 1], got {value}")
        effective_context_prob = partial_context_prob if num_context_chunks > 1 else 0.0
        if init_schedule_prob + effective_context_prob > 1.0:
            raise ValueError(
                f"init_schedule_prob + partial_context_prob must be <= 1, got "
                f"{init_schedule_prob} + {partial_context_prob}"
            )
        if not 0.0 <= obs_offset_prob <= 1.0:
            raise ValueError(f"`obs_offset_prob` must be in [0, 1], got {obs_offset_prob}")
        if obs_offset_range < 0:
            raise ValueError(f"`obs_offset_range` must be >= 0, got {obs_offset_range}")
        if obs_offset_prob > 0 and obs_offset_range == 0:
            raise ValueError("`obs_offset_prob` > 0 requires `obs_offset_range` >= 1")
        if num_context_chunks > 1 and partial_context_prob <= 0:
            logger.warning(
                "num_context_chunks=%d with partial_context_prob=0: inference will visit "
                "context sizes h=1..%d while the cache fills, but training never shows them. "
                "Set rolling.partial_context_prob > 0.",
                num_context_chunks, num_context_chunks - 1,
            )
        elif num_context_chunks <= 1 and partial_context_prob > 0:
            logger.warning(
                "partial_context_prob=%s has no effect with num_context_chunks=%d; "
                "boundary init covers h=0 and steady rolling covers h=%d.",
                partial_context_prob, num_context_chunks, num_context_chunks,
            )

        self.window_blocks = int(window_blocks)
        self.num_context_chunks = int(num_context_chunks)
        self.chunk_latents = int(chunk_latents)
        self.actions_per_chunk = int(actions_per_chunk)
        self.init_schedule_prob = float(init_schedule_prob)
        self.partial_context_prob = float(partial_context_prob)
        self.obs_offset_prob = float(obs_offset_prob)
        self.obs_offset_range = int(obs_offset_range) if obs_offset_prob > 0 else 0
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
        "init_schedule_prob", "partial_context_prob", "obs_offset_prob", "obs_offset_range",
    )

    def get_rolling_config(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.ROLLING_KEYS}

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "rolling": self.get_rolling_config(),
            "scheduler": {
                "shift": self.train_video_scheduler.shift,
                "num_train_timesteps": self.train_video_scheduler.num_train_timesteps,
            },
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = super().load_checkpoint(path, optimizer=optimizer)

        saved_sched = payload.get("scheduler")
        if saved_sched is not None:
            cur = self.train_video_scheduler
            if (saved_sched.get("shift") != cur.shift
                    or saved_sched.get("num_train_timesteps") != cur.num_train_timesteps):
                raise ValueError(
                    f"Scheduler mismatch: checkpoint trained with {saved_sched}, yaml has "
                    f"shift={cur.shift}, num_train_timesteps={cur.num_train_timesteps}. "
                    "The rolling schedule is defined by the shift; fix the scheduler yaml."
                )

        saved = payload.get("rolling")
        if saved is None:
            logger.warning(
                "Checkpoint has no rolling config (legacy/base checkpoint); keeping the "
                "current yaml rolling settings — verify they match how it was trained."
            )
            return payload

        expected_keys = set(self.ROLLING_KEYS)
        if not isinstance(saved, dict) or not set(saved) <= expected_keys:
            got = sorted(saved) if isinstance(saved, dict) else type(saved).__name__
            raise ValueError(
                "Checkpoint rolling config keys do not match: "
                f"expected={sorted(expected_keys)}, got={got}."
            )
        missing = expected_keys - set(saved)
        if missing:
            logger.warning(
                "Checkpoint rolling config predates %s; keeping the current yaml values for them.",
                sorted(missing),
            )
            saved = {**{key: getattr(self, key) for key in missing}, **saved}
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

    def _rolling_ladder(
        self, num_inference_steps: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """S-rung schedule from the video inference scheduler.
        Returns (timesteps [S], deltas [S]); rung 0 = noisiest, rung S-1 = cleanest."""
        return self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------ mask

    @torch.no_grad()
    def _build_rolling_attention_mask(
        self,
        ctx_chunks: int,
        win_chunks: int,
        tokens_per_frame: int,
        actions_per_chunk: int,
        device: torch.device,
    ) -> torch.Tensor:
        nfpb = self.chunk_latents
        video_frames = 1 + (ctx_chunks + win_chunks) * nfpb
        video_seq_len = video_frames * tokens_per_frame
        action_seq_len = win_chunks * actions_per_chunk
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

        a_chunk = torch.arange(win_chunks, device=device).repeat_interleave(actions_per_chunk)
        a = slice(video_seq_len, total)

        mask[a, v] = True

        same_chunk = a_chunk.view(-1, 1) == a_chunk.view(1, -1)
        mask[a, a] = same_chunk
        return mask

    @torch.no_grad()
    def _build_training_attention_mask(
        self,
        ctx_chunks: torch.Tensor,
        tokens_per_frame: int,
        actions_per_chunk: int,
        device: torch.device,
    ) -> torch.Tensor:
        if ctx_chunks.ndim != 1:
            raise ValueError(f"`ctx_chunks` must be 1D [B], got {tuple(ctx_chunks.shape)}")

        H, W = self.num_context_chunks, self.window_blocks
        max_video_len = (1 + (H + W) * self.chunk_latents) * tokens_per_frame
        max_action_len = W * actions_per_chunk
        total_len = max_video_len + max_action_len

        masks = {}
        h_values = [int(value) for value in ctx_chunks.tolist()]
        for h in set(h_values):
            if not 0 <= h <= H:
                raise ValueError(f"Context chunks must be in [0, {H}], got {h}")
            video_len = (1 + (h + W) * self.chunk_latents) * tokens_per_frame
            action_len = W * actions_per_chunk
            compact = self._build_rolling_attention_mask(
                ctx_chunks=h,
                win_chunks=W,
                tokens_per_frame=tokens_per_frame,
                actions_per_chunk=actions_per_chunk,
                device=device,
            )
            mask = torch.eye(total_len, dtype=torch.bool, device=device)
            mask[:video_len, :video_len] = compact[:video_len, :video_len]
            mask[:video_len, max_video_len : max_video_len + action_len] = compact[
                :video_len, video_len:
            ]
            mask[max_video_len : max_video_len + action_len, :video_len] = compact[
                video_len:, :video_len
            ]
            mask[
                max_video_len : max_video_len + action_len,
                max_video_len : max_video_len + action_len,
            ] = compact[video_len:, video_len:]
            masks[h] = mask

        if len(masks) == 1:
            return masks[h_values[0]]
        return torch.stack([masks[h] for h in h_values])

    # ------------------------------------------------------------------ training

    def _sample_training_layouts(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample history length and boundary-init schedule per example."""
        if batch_size < 1:
            raise ValueError(f"`batch_size` must be >= 1, got {batch_size}")
        H = self.num_context_chunks
        p_i = self.init_schedule_prob
        p_c = self.partial_context_prob if H > 1 else 0.0
        r = torch.rand((batch_size,))
        init = r < p_i
        context_fill = (r >= p_i) & (r < p_i + p_c)
        h = torch.full((batch_size,), H, dtype=torch.long)
        h[init] = 0
        count = int(context_fill.sum().item())
        if count > 0:
            h[context_fill] = torch.randint(1, H, (count,))
        return h, init

    def _apply_obs_offset(
        self,
        sample: dict,
        h: torch.Tensor,
        tiled: bool,
        allow_offset: torch.Tensor,
    ):
        """Execution-deviation augmentation. The data carries `obs_offset_range` extra video
        frames of margin on each side; the center crop is the GT clip and supplies
        conditioning, context, and supervision. With probability `obs_offset_prob`, the
        window's noising base is instead taken from a clip shifted by a per-sample offset
        in [-range, range] video frames, so the model learns to steer window content toward
        what the latest observation says. Returns (sample sliced to the GT clip, dev|None)."""
        m = self.obs_offset_range
        if m == 0:
            return sample, None

        video, action = sample["video"], sample["action"]
        batch_size = video.shape[0]
        if h.shape != (batch_size,) or allow_offset.shape != (batch_size,):
            raise ValueError("`h` and `allow_offset` must be 1D tensors matching the batch size.")
        nfpb, aspc = self.chunk_latents, self.actions_per_chunk
        win = self.window_blocks
        tdf = int(self.vae.temporal_downsample_factor)
        frames = 1 + tdf * nfpb * (self.num_context_chunks + self.window_blocks)
        if video.shape[2] != frames + 2 * m:
            raise ValueError(
                f"obs_offset_range={m} expects {frames + 2 * m} video frames "
                f"({frames} + 2*{m} margin), got {video.shape[2]}. Set data num_frames accordingly."
            )
        if action.shape[1] % (video.shape[2] - 1) != 0:
            raise ValueError(
                f"Action horizon ({action.shape[1]}) must be divisible by video transitions ({video.shape[2] - 1})."
            )
        apt = action.shape[1] // (video.shape[2] - 1)
        if apt * tdf * nfpb != aspc:
            raise ValueError(
                f"Data supplies {apt * tdf * nfpb} actions per chunk but `actions_per_chunk={aspc}`."
            )

        image_is_pad = sample.get("image_is_pad", None)
        action_is_pad = sample.get("action_is_pad", None)
        sample = dict(sample)
        sample["video"] = video[:, :, m : m + frames]
        sample["action"] = action[:, m * apt : (m + frames - 1) * apt]
        if image_is_pad is not None:
            sample["image_is_pad"] = image_is_pad[:, m : m + frames]
        if action_is_pad is not None:
            sample["action_is_pad"] = action_is_pad[:, m * apt : (m + frames - 1) * apt]
        if sample.get("proprio", None) is not None:
            sample["proprio"] = sample["proprio"][:, m * apt : (m + frames - 1) * apt]

        if not self.dit.training or self.obs_offset_prob <= 0:
            return sample, None

        active = allow_offset.to(device=video.device, dtype=torch.bool)
        active &= torch.rand(batch_size, device=video.device) < self.obs_offset_prob
        if not active.any().item():
            return sample, None

        # never shift into padded future: pads are a contiguous episode tail, so cap the
        # positive offsets per sample at the last fully valid deviated clip
        o_hi = torch.full((batch_size,), m, dtype=torch.long, device=video.device)
        if image_is_pad is not None:
            o_hi = torch.minimum(o_hi, (~image_is_pad).sum(dim=1) - (m + frames))
        if action_is_pad is not None:
            o_hi = torch.minimum(o_hi, (~action_is_pad).sum(dim=1) // apt - (m + frames - 1))
        o_hi = o_hi.clamp(min=-m)

        width = (o_hi + m + 1).to(torch.float32)
        offset = (torch.rand(batch_size, device=video.device) * width).long() - m
        offset = torch.where(active, offset, torch.zeros_like(offset))
        h_data = h.to(device=video.device)
        f_idx = m + offset.view(-1, 1) + torch.arange(frames, device=video.device).view(1, -1)
        a_idx = (
            (m + offset).view(-1, 1) * apt
            + h_data.view(-1, 1) * aspc
            + torch.arange(win * aspc, device=video.device).view(1, -1)
        )

        dev_video = torch.gather(
            video, 2,
            f_idx.view(batch_size, 1, frames, 1, 1).expand(-1, video.shape[1], -1, video.shape[3], video.shape[4]),
        )
        dev_latents = self._encode_video_latents(
            dev_video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True), tiled=tiled
        )
        dev_action = torch.gather(action, 1, a_idx.unsqueeze(-1).expand(-1, -1, action.shape[2]))

        latent_idx = (
            1
            + h.to(device=dev_latents.device).view(-1, 1) * nfpb
            + torch.arange(win * nfpb, device=dev_latents.device).view(1, -1)
        )
        dev_window_latents = torch.gather(
            dev_latents,
            2,
            latent_idx.view(batch_size, 1, -1, 1, 1).expand(
                -1, dev_latents.shape[1], -1, dev_latents.shape[3], dev_latents.shape[4]
            ),
        )

        dev_latent_is_pad = dev_action_is_pad = None
        if image_is_pad is not None:
            dev_frame_is_pad = torch.gather(image_is_pad, 1, f_idx)
            dev_all_latent_is_pad = dev_frame_is_pad[:, 1:].view(batch_size, -1, tdf).all(dim=2)
            dev_latent_is_pad = torch.gather(
                dev_all_latent_is_pad,
                1,
                (latent_idx - 1).to(dev_all_latent_is_pad.device),
            ).to(self.device)
        if action_is_pad is not None:
            dev_action_is_pad = torch.gather(action_is_pad, 1, a_idx).to(self.device)

        return sample, {
            "active": active.to(self.device),
            "window_latents": dev_window_latents,
            "window_action": dev_action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            "window_latent_is_pad": dev_latent_is_pad,
            "window_action_is_pad": dev_action_is_pad,
        }

    def training_loss(self, sample, tiled: bool = False):
        video = sample.get("video")
        if not isinstance(video, torch.Tensor) or video.ndim < 1:
            raise ValueError("`sample['video']` must be a batched tensor.")
        batch_size = int(video.shape[0])
        H, W = self.num_context_chunks, self.window_blocks
        h_cpu, init_cpu = self._sample_training_layouts(batch_size)

        sample, dev = self._apply_obs_offset(
            sample,
            h=h_cpu,
            tiled=tiled,
            allow_offset=~init_cpu,
        )

        if sample.get("proprio", None) is not None:
            proprio = sample["proprio"]
            aspc = self.actions_per_chunk
            proprio_idx = h_cpu.to(proprio.device) * aspc
            max_proprio_idx = int(proprio_idx.max().item())
            if max_proprio_idx >= proprio.shape[1]:
                raise ValueError(
                    f"proprio has {proprio.shape[1]} steps; window start index "
                    f"{max_proprio_idx} is out of range"
                )
            sample = dict(sample)
            sample["proprio"] = torch.gather(
                proprio,
                1,
                proprio_idx.view(-1, 1, 1).expand(-1, 1, proprio.shape[2]),
            )

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

        device, dtype = input_latents.device, input_latents.dtype
        h = h_cpu.to(device=device)
        init = init_cpu.to(device=device)
        has_structural_padding = bool((h_cpu < H).any().item())
        n_steps = float(self.train_video_scheduler.num_train_timesteps)

        t_global = torch.rand((batch_size, 1), device=device)
        slots = torch.arange(W, device=device, dtype=torch.float32).view(1, W)
        u_steady = (slots + t_global) / W
        u_init = (slots / W + t_global).clamp(max=1.0)
        u_chunk = torch.where(init.view(-1, 1), u_init, u_steady)
        sch = self.train_video_scheduler
        sigma_chunk = sch._phi(u_chunk, sch.shift)
        t_window_chunk = sigma_chunk * n_steps

        window_latent_idx = (
            1
            + h.view(-1, 1) * nfpb
            + torch.arange(W * nfpb, device=device).view(1, -1)
        )
        latent_gather_idx = window_latent_idx.view(batch_size, 1, -1, 1, 1).expand(
            -1, input_latents.shape[1], -1, input_latents.shape[3], input_latents.shape[4]
        )
        window_latents = torch.gather(input_latents, 2, latent_gather_idx)

        base_window_latents = window_latents
        if dev is not None:
            base_window_latents = torch.where(
                dev["active"].view(batch_size, 1, 1, 1, 1),
                dev["window_latents"],
                window_latents,
            )

        noise_video = torch.randn_like(window_latents)
        t_window_frame = t_window_chunk.repeat_interleave(nfpb, dim=1)
        sigma_video = (t_window_frame / n_steps).to(dtype).view(batch_size, 1, -1, 1, 1)
        noisy_window_latents = (
            (1 - sigma_video) * base_window_latents + sigma_video * noise_video
        )
        target_video = noise_video - window_latents

        latent_position = torch.arange(latent_t, device=device).view(1, -1)
        active_latent_len = 1 + (h + W) * nfpb
        padded_latents = input_latents
        if has_structural_padding:
            active_latent = latent_position < active_latent_len.view(-1, 1)
            padded_latents = torch.where(
                active_latent.view(batch_size, 1, latent_t, 1, 1),
                input_latents,
                torch.randn_like(input_latents),
            )
        noisy_latents = padded_latents.scatter(2, latent_gather_idx, noisy_window_latents)

        context_latent_len = 1 + h * nfpb
        timestep_video = torch.full(
            (batch_size, latent_t), n_steps, dtype=torch.float32, device=device
        )
        timestep_video = torch.where(
            latent_position < context_latent_len.view(-1, 1),
            torch.zeros_like(timestep_video),
            timestep_video,
        ).scatter(1, window_latent_idx, t_window_frame)

        window_action_idx = (
            h.view(-1, 1) * aspc
            + torch.arange(W * aspc, device=device).view(1, -1)
        )
        action_gather_idx = window_action_idx.unsqueeze(-1).expand(-1, -1, action.shape[2])
        window_action = torch.gather(action, 1, action_gather_idx)
        window_action_is_pad = (
            None if action_is_pad is None else torch.gather(action_is_pad, 1, window_action_idx)
        )

        t_window_action = t_window_chunk.repeat_interleave(aspc, dim=1)
        base_action = window_action
        if dev is not None:
            base_action = torch.where(
                dev["active"].view(batch_size, 1, 1),
                dev["window_action"],
                window_action,
            )
        noise_action = torch.randn_like(window_action)
        sigma_action = (t_window_action / n_steps).to(dtype).unsqueeze(-1)
        noisy_window_action = (1 - sigma_action) * base_action + sigma_action * noise_action
        target_action = noise_action - window_action

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

        attention_mask = self._build_training_attention_mask(
            ctx_chunks=h_cpu,
            tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            actions_per_chunk=aspc,
            device=device,
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

        pred_w = torch.gather(pred_video, 2, latent_gather_idx)
        video_loss_frame = torch.nn.functional.mse_loss(
            pred_w.float(), target_video.float(), reduction="none"
        ).mean(dim=(1, 3, 4))

        video_weight = self.train_video_scheduler.training_weight(
            t_window_frame.reshape(-1)
        ).reshape(batch_size, -1).to(video_loss_frame.device, dtype=video_loss_frame.dtype)
        video_weight = video_weight.clamp(min=0.05)

        if image_is_pad is not None:
            factor = int(self.vae.temporal_downsample_factor)
            latent_is_pad = image_is_pad[:, 1:].view(batch_size, -1, factor).all(dim=2)
            window_valid = (~torch.gather(latent_is_pad, 1, window_latent_idx - 1)).to(
                video_loss_frame.dtype
            )
        else:
            window_valid = torch.ones_like(video_loss_frame)
        if dev is not None and dev["window_latent_is_pad"] is not None:
            window_valid = window_valid * (~dev["window_latent_is_pad"]).to(window_valid.dtype)
        valid_sum = window_valid.sum(dim=1).clamp(min=1.0)
        schedule_active = (~init.view(-1, 1)) | (u_chunk < 1.0)
        video_loss_mask = window_valid * schedule_active.repeat_interleave(
            nfpb, dim=1
        ).to(window_valid.dtype)

        loss_video = (
            (video_loss_frame * video_weight * video_loss_mask).sum(dim=1) / valid_sum
        ).mean()

        action_loss_token = torch.nn.functional.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)

        action_weight = self.train_action_scheduler.training_weight(
            t_window_action.reshape(-1)
        ).reshape(batch_size, -1).to(action_loss_token.device, dtype=action_loss_token.dtype)
        action_weight = action_weight.clamp(min=0.05)

        if window_action_is_pad is not None:
            action_valid = (~window_action_is_pad).to(action_loss_token.dtype)
        else:
            action_valid = torch.ones_like(action_loss_token)
        if dev is not None and dev["window_action_is_pad"] is not None:
            action_valid = action_valid * (~dev["window_action_is_pad"]).to(action_valid.dtype)
        valid_sum = action_valid.sum(dim=1).clamp(min=1.0)
        action_loss_mask = action_valid * schedule_active.repeat_interleave(
            aspc, dim=1
        ).to(action_valid.dtype)

        loss_action = (
            (action_loss_token * action_weight * action_loss_mask).sum(dim=1) / valid_sum
        ).mean()

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
        self._stream_steps: Optional[int] = None               # S locked for the episode
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
        advance: Optional[list] = None,
    ):
        """One denoise pass over the window against clean context; advances every rung by 1.
        `advance[j]=False` freezes chunk j (boundary init phase: clipped chunks stay pure noise)."""
        nfpb = self.chunk_latents
        win = len(self._rungs)
        h = (ctx_latents.shape[2] - 1) // nfpb

        latents = torch.cat([ctx_latents, self._window_latents], dim=2)
        rungs = torch.tensor(self._rungs, device=self.device)
        t_chunk = ladder_t[rungs].to(torch.float32)                              # [win]
        timestep_video = torch.zeros((1, latents.shape[2]), dtype=torch.float32, device=self.device)
        timestep_video[0, ctx_latents.shape[2]:] = t_chunk.repeat_interleave(nfpb)
        t_window_action = t_chunk.repeat_interleave(aspc).unsqueeze(0)           # [1, win*aspc]

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
                action_tokens=self._window_action,
                timestep=t_window_action,
                context=ctx,
                context_mask=ctx_mask,
            )
            attention_mask = self._build_rolling_attention_mask(
                ctx_chunks=h,
                win_chunks=win,
                tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                actions_per_chunk=aspc,
                device=self.device,
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

        adv = [True] * win if advance is None else advance
        pred_window = pred_video[:, :, ctx_latents.shape[2]:]
        for j, r in enumerate(self._rungs):
            if not adv[j]:
                continue
            delta = ladder_delta[r]
            vs = slice(j * nfpb, (j + 1) * nfpb)
            self._window_latents[:, :, vs] += pred_window[:, :, vs] * delta
            as_ = slice(j * aspc, (j + 1) * aspc)
            self._window_action[:, as_] += pred_action[:, as_] * delta
        self._rungs = [r + 1 if a else r for r, a in zip(self._rungs, adv)]

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
        num_inference_steps: Optional[int] = None,
    ) -> dict[str, Any]:
        """One control step: feed newly observed frames, get the next action chunk.

        First call: `new_frames` [1, 3, 1, H, W] (single observation) — runs the boundary
        init phase (full window denoised from pure noise into the rolling state).
        Later calls: [1, 3, 4*chunk_latents, H, W] (the frames observed while executing
        the previous chunk). Returns {'action': [aspc, action_dim], 'video': predicted
        front chunk latents}.
        """
        self.eval()
        if new_frames.ndim != 5 or new_frames.shape[0] != 1 or new_frames.shape[1] != 3:
            raise ValueError(f"`new_frames` must be [1, 3, T, H, W], got {tuple(new_frames.shape)}")

        nfpb = self.chunk_latents
        H, W = self.num_context_chunks, self.window_blocks
        if num_inference_steps is None:
            raise ValueError("`num_inference_steps` is required (inference-only knob, not model config)")
        S = int(num_inference_steps)
        if S % W != 0:
            raise ValueError(f"num_inference_steps ({S}) must be divisible by window_blocks ({W})")
        if self._stream_steps is None:
            self._stream_steps = S
        elif S != self._stream_steps:
            raise ValueError(f"num_inference_steps changed mid-stream: {self._stream_steps} -> {S}")
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
        ladder_t, ladder_delta = self._rolling_ladder(S, self.device, torch.float32)

        if first_call:
            # boundary init phase (t^init): full window of pure noise; chunk j stays
            # clipped at sigma=1 until pass j*sub, then advances one rung per pass
            for _ in range(W):
                self._push_noise_chunk(z, latent_h, latent_w, aspc, seed)
            for p in range(S):
                self._window_pass(
                    ctx_latents, context, context_mask, ladder_t, ladder_delta, aspc,
                    text_cfg_scale, negative_context, negative_context_mask,
                    advance=[p >= j * sub for j in range(W)],
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
        del action, sigma_shift, rand_device, tiled
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
                num_inference_steps=num_inference_steps,
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
