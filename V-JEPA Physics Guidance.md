# TITAN-style V-JEPA physics guidance

The default physics objective now evaluates a predicted final video instead of
decoding the current noisy latent directly.

At a selected callback step, CogVideoX has already produced the current
`z_(t-1)`. For every sampled latent direction `v`, the guide computes

```text
z_plus  = z_(t-1) + epsilon * v
z_minus = z_(t-1) - epsilon * v
```

Both branches are independently denoised over all remaining timesteps. The
resulting final videos are evaluated with the same V-JEPA pixel probes:

```text
dE/dv ~= (E(final(z_plus)) - E(final(z_minus))) / (2 * epsilon)
```

The directional estimates are averaged and applied to the main-path
`z_(t-1)`. Temporary rollout latents are discarded. The main scheduler is never
used by a temporary rollout; every branch receives a fresh scheduler instance.

This is a TITAN-style final-rollout objective implemented with central finite
differences. It does not yet use TITAN-Guide's forward-mode AD/JVP, so one
latent direction requires two remaining-step rollouts.

## Minimal first test

Use one guidance step and one latent direction first:

```bash
python inference/cli_demo.py \
  --prompt "A person throws a large amount of water directly toward a burning pile of wooden logs." \
  --model_path THUDM/CogVideoX-2b \
  --generate_type t2v \
  --output_path outputs/titan_physics_guidance.mp4 \
  --num_inference_steps 50 \
  --num_frames 49 \
  --guidance_scale 6.0 \
  --seed 42 \
  --dtype float16 \
  --fps 8 \
  --physics_guidance_scale 0.01 \
  --physics_guidance_start_step 20 \
  --physics_guidance_end_step 20 \
  --physics_guidance_interval 1 \
  --physics_guidance_latent_epsilon 0.05 \
  --physics_guidance_latent_directions 1 \
  --physics_guidance_directions 1 \
  --physics_guidance_device cuda
```

With 50 denoising steps, guidance after step 20 rolls each branch through the
remaining 29 steps. One latent direction therefore performs 58 additional
transformer steps, plus two final VAE decodes and V-JEPA energy evaluations.

The log should contain:

```text
objective=final_rollout
positive_energy=...
negative_energy=...
derivative_abs_mean=...
relative_update=...
```

To reproduce the old current-latent decode as an ablation, add:

```text
--physics_guidance_current_decode
```

## Current approximation

The main generation uses dynamic classifier-free guidance. A temporary rollout
freezes classifier-free guidance at the value active at the branching step.
This avoids restarting the dynamic schedule with an incorrect shortened-step
index. The plus and minus branches use the same frozen value, so their energy
difference remains paired.