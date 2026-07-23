# V-JEPA Physics Guidance

This experimental guide corrects CogVideoX latents during denoising by
minimizing a finite-difference approximation of V-JEPA 2 feature sensitivity.
It does not add a text description of the expected physical outcome.

## Baseline

Keep `--physics_guidance_scale 0` (the default). The original CogVideoX
pipeline is unchanged.

## Guided generation

Start with a small correction strength and only three guide evaluations:

```bat
python inference/cli_demo.py ^
  --prompt "A small fire burns on a compact pile of wooden logs in an outdoor fire pit. From the side of the frame, a person quickly swings a bucket and throws a large amount of clear water directly onto the flames in one sudden motion. The water forms a wide, forceful splash that covers the fire and the wooden logs. The camera remains stationary, keeping the fire pit and the splash of water clearly visible throughout the scene." ^
  --model_path zai-org/CogVideoX-2b ^
  --generate_type t2v ^
  --output_path outputs/fire_water_guided.mp4 ^
  --num_inference_steps 50 ^
  --num_frames 49 ^
  --guidance_scale 6.0 ^
  --seed 42 ^
  --dtype float16 ^
  --fps 8 ^
  --physics_guidance_scale 0.01 ^
  --physics_guidance_start_step 20 ^
  --physics_guidance_end_step 40 ^
  --physics_guidance_interval 10 ^
  --physics_guidance_frames 16 ^
  --physics_guidance_epsilon 0.01
```

This applies corrections at zero-based denoising steps 20, 30, and 40.
Compare against a baseline made with the same prompt, seed, scheduler, and
generation settings.

The first run also downloads
`facebook/vjepa2-vitl-fpc64-256`. V-JEPA backpropagation and VAE decoding make
guided generation substantially slower and more memory intensive than the
baseline.

## Main options

- `--physics_guidance_scale`: normalized latent correction strength. Try
  `0.005`, `0.01`, then `0.02`.
- `--physics_guidance_interval`: number of denoising steps between guide
  evaluations.
- `--physics_guidance_start_step`, `--physics_guidance_end_step`: inclusive
  step range.
- `--physics_guidance_frames`: uniformly sampled decoded frames passed to
  V-JEPA.
- `--physics_guidance_epsilon`: pixel-space finite-difference magnitude.
- `--physics_guidance_device`: `auto`, `cuda`, or `cpu`.

