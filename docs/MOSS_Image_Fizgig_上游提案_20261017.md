# Proposal: MOSS-Image-Base image LoRA family

I maintain a Windows LoRA training frontend that integrates Fizgig. I would like to discuss a MOSS-Image family before starting implementation, as requested by the driver documentation. This is a scope proposal, not a tested driver or a claim of working training support.

## Model and reference sources

- Project / official code: https://github.com/OpenMOSS/MOSS-Image
- Project page: https://openmoss.ai/MOSS-Image/
- Base weights and standalone reference pipeline: https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base
- Official domestic mirror: https://modelscope.cn/models/openmoss/MOSS-Image-Base
- Official training notes: https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE_ZH.md#lora
- Official ComfyUI plugin: https://github.com/OpenMOSS/MOSS-Image/tree/main/comfyui/moss-image

Sources inspected on 2026-10-17: MOSS repository `c1355e6b1986dfe8fba3c0455783a05899212342`, Base model repository `1ceede80badddd6a7812069619edaa5633587192`; Fizgig driver documentation at `4847537845aa7fb6ca7961a0d0546cb2befcdcbb`.

The Base model card declares Apache-2.0. Component/code notices would be preserved and checked individually before vendoring anything.

## What I found in the reference

- A custom ~5B DiT: 42 blocks, width 3072, text dimension 4096; this is not a renamed Flux family.
- A Qwen3.5 text encoder (the official ComfyUI plugin identifies the 9B variant), with its exact chat template, final hidden states and attention mask. The official guide explicitly requires keeping the text encoder frozen during post-training.
- `AutoencoderKLFlux2`; its config has 32 latent channels and 2x2 packing, while the transformer consumes 128 channels. VAE packing/normalisation needs reference parity, not assumptions copied from another family.
- A custom `MOSSImageScheduler`, with shifted sigmas snapped to a training grid. The reference pipeline uses 50 steps, CFG 3 and shift 5, and defaults to `uncond_skip_layers=[22]`. These details need parity checks before preview defaults are established.
- The public guide discusses LoRA/post-training and captioning, but I have not found a complete public LoRA training recipe in the inspected sources. I will not infer the training objective, timestep distribution, learning rate or VRAM figures solely from inference defaults.

## Proposed first milestone, subject to your scope guidance

An experimental hidden `moss_image` still-image family for **MOSS-Image-Base**, initially aiming for ordinary image LoRA with a frozen text encoder, cached text/latents, epoch previews, checkpoint export, and proper pause/resume. SFT could be considered after verifying architecture/key compatibility.

The implementation would follow the current description + driver + model-package interface and reference the official implementation. It would stay hidden until numerical parity, short real training, ComfyUI LoRA key/output checks, offline loading, and measured small-card training/preview memory meet your checklist. Optional capabilities would be evaluated against ABILITIES.md and enabled only after validation, with limitations documented.

In particular, the standalone pipeline and official ComfyUI implementation use different internal projection layouts in places. LoRA target selection and saved key mapping need to cover the corresponding ComfyUI modules, including any fused/split projection conversion. A run finishing is not evidence of export compatibility.

## Questions before implementation

1. Is someone already working on this model, and would an experimental Base image-LoRA family be welcome?
2. Would you prefer a vendored official model implementation or a driver wrapping the pinned standalone Diffusers components? Is an existing family the best reference for this scheduler/conditioning combination?
3. Do you know of an authoritative training recipe or contact with the model authors for the loss/timestep distribution and suggested LoRA targets? I would rather resolve these than guess from the sampler.
4. The Qwen3.5 encoder may dominate the caching-stage memory. Do you have a preferred supported quantised/offload path for it? I am not claiming an 8 GB minimum or AMD compatibility before real runs.

No driver has been implemented or GPU-validated yet. I am opening this discussion first to agree the scope and avoid duplicating work.
