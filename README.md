# RRCE

Standard RRCE for nudity, dog, and Van Gogh erasure with Stable Diffusion v1.4.

RRCE alternates an anchor-guided erasure update with a preservation update. The two objectives keep independent adaptive AdamW states. Between defense blocks, textual inversion discovers a recovered concept against the current model; the next defense block erases that concept using the same frozen pretrained teacher.

## Installation

Use Python 3.10 or later and a CUDA-enabled PyTorch installation compatible with the machine. Install dependencies from the repository root:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

Pretrained models, datasets, generated images, and checkpoints are external to this repository. Full-model training holds the student, teacher, and two optimizer histories in memory; choose a GPU with sufficient memory. Integration tests use small randomly initialized Diffusers models and require no downloads.

## Layout

```text
train.py                         Training entry point
prepare_data.py                  Manifests, retain latents, CCE galleries
generate.py                      Image generation from a CSV manifest
evaluate.py                      Main experiment evaluation
configs/
  base.json                      Shared training and evaluation settings
  nudity.json                    Nudity experiment overrides
  dog.json                       Dog experiment overrides
  van_gogh.json                  Style experiment overrides
  concepts/                      Target, retain, anchor, inversion templates
  protocol/                      Evaluation prompts and classification labels
rrce/
  config.py                      Typed configuration loading and validation
  models/                        Diffusion model and parameter selection
  data/                          Datasets, balanced streams, manifests, assets
  optim/                         Adaptive AdamW and independent objective states
  training/                      Trainer, objectives, recovery, checkpoints
  attacks/                       CCE and UnlearnDiff
  evaluation/                    Sampling, detection, retention, quality
experiments/                     Bash and Slurm task launchers
tests/                           Numerical and small-model integration tests
```

## Configuration

Each task inherits `configs/base.json` and specifies only its differences. Concept lists and prompt collections are separate data files. Run commands from the repository root: relative data and output paths use the working directory. Configuration inheritance and referenced concept files use the selected configuration file's directory.

Inspect all settings without loading a model:

```bash
python train.py --config configs/dog.json --print-config
```

Override an existing setting with `--set KEY=VALUE`. Values use JSON types; ordinary path strings need no JSON quotes. Repeat the option for multiple settings. Unknown keys and invalid values fail before model loading.

```bash
python train.py --config configs/dog.json \
  --set training.optimizer.learning_rate=0.00002 \
  --set data.target_images=data/dog/target \
  --set training.output=outputs/dog/train
```

Every run saves its resolved configuration, including concept lists, as `config.json`. This file can itself be supplied to `--config`.

| Group | Controls |
| --- | --- |
| `runtime` | Device, training precision, seed, data workers, logging |
| `model` | Diffusion checkpoint, training text encoder, resolution |
| `data` | Training and evaluation input paths |
| `training` | Defense blocks, ordered update pairs, guidance, parameter scope, optimizer |
| `recovery` | Textual inversion steps, optimizer, image preprocessing, token prefix |
| `assets` | Retain latent seeds and asset generation settings |
| `evaluation` | Counts, per-split sampling, detectors, CLIP, FID |
| `cce` | Gallery size, learned token, inversion budget, sampling |
| `ud` | Prefix length, timesteps, optimization budget, sampling |
| `protocol` | Prompt selection and deterministic manifest seeds |

## Data preparation

Provide the inputs needed for the experiments being run:

```text
data/
  dog/target/                     Dog training images
  nudity/target/                  Nudity training images
  van_gogh/target/                Style training images
  i2p.csv                        prompt,nudity_percentage
  coco30k.csv                     case_number,prompt,evaluation_seed,coco_id
  coco_reference/                Matching 30,000 real COCO images
  models/nudenet.onnx             NudeNet detector
  models/wikiart/                 WikiArt model and image processor
```

Target training images must be separate from evaluation images. Recovery partitions the sorted target-image list into consecutive subsets, one per recovery round. Dog and Van Gogh manifest preparation does not require I2P. COCO uses the supplied fixed 30,000-case manifest; arbitrary captions or different reference images do not reproduce the same FID protocol.

Prompt manifests contain `sample_id,prompt,seed`; retain evaluation also includes `target_label`. Generated manifests add `image_path`, relative to their containing directory. Retain training manifests contain `prompt,image_path`, with one image or scaled diffusion latent tensor per row. Retain training uses the 20 bare labels plus the empty prompt, without evaluation templates.

```bash
bash experiments/dog.sh manifests
bash experiments/dog.sh retain
bash experiments/dog.sh gallery
```

Replace `dog.sh` with `nudity.sh` or `van_gogh.sh` for the other experiments. Retain generation caches eight diffusion latents per concept using seeds 0–7, DDIM, 50 steps, and guidance 3. Gallery generation uses the base model; the nudity gallery filters images with NudeNet. Existing asset directories and manifests are not overwritten.

## Training

```bash
bash experiments/dog.sh train
```

Standard settings are three defense blocks of 500 ordered erase/retain pairs, with two 3,000-step recovery rounds. Each pair applies the erasure optimizer first, then computes and applies the retain update to the updated student. Both optimizer histories persist across defense blocks.

| Setting | Standard value |
| --- | --- |
| Teacher | Frozen pretrained U-Net throughout training |
| Training text encoder | `openai/clip-vit-large-patch14` |
| U-Net learning rate | `2e-5` |
| Adaptive first momentum | `[0.1, 0.9]` |
| Fixed second momentum | `0.999` |
| Guidance scale / projection weight | `4.0 / 1.0` |
| Recovery learning rate / weight decay | `0.005 / 0.01` |
| Recovery timestep range | `[0, 999)` |
| Dog and nudity scope | All selected linear and convolution layers |
| Van Gogh scope | Selected cross-attention linear layers |

`full` follows the training implementation's layer selection: linear and convolution modules are trainable; normalization parameters remain frozen. Selected parameter names are saved for inspection. Recovery freezes the U-Net and updates only the new token embedding; previous embedding rows are restored after every step.

Training writes:

- `unet.pt`: complete U-Net state dictionary for inference.
- `recovery_1.pt`, `recovery_2.pt`, `recovery_tokens.pt`: learned token embeddings.
- `latest.pt`: latest completed defense-block checkpoint, including optimizer histories, tokens, and random states.
- `training.jsonl`: losses, gradient cosines, adaptive momentum values.
- `config.json`, `trainable_parameters.json`, `status.json`: settings and progress.

Resume with the same resolved configuration and output directory:

```bash
python train.py --config outputs/dog/train/config.json \
  --resume outputs/dog/train/latest.pt
```

Resume starts after the last saved defense block. An interrupted recovery or defense block is repeated. Checkpoints can be large because they preserve both optimizer histories.

## Evaluation

```bash
bash experiments/dog.sh evaluate
```

Evaluation writes `results.json`, per-split `metrics.json`, and `per_image.csv`. CCE also saves its learned embedding; UnlearnDiff saves attack token IDs and base-model reference images. Results contain only the splits requested in that invocation.

| Task | Erased images | UD prompts | CCE test images | CCE gallery |
| --- | ---: | ---: | ---: | --- |
| Nudity | 440 | 95 | 95 | Disjoint I2P prompts, NudeNet-filtered images |
| Dog | 500 | 50 | 500 | 30 images |
| Van Gogh | 500 | 50 | 500 | 6 images |

Each task also evaluates 500 retain images and 30,000 COCO images. Van Gogh CCE follows the six-image main-text protocol. A 30-image gallery has a different evaluation budget and does not represent the same run.

- **Erased, UD, CCE:** target-detection percentage, lower is better. Nudity uses six configured exposed-body labels at threshold 0.75. Dog uses Faster R-CNN ResNet-50 FPN v2 at threshold 0.5. Style uses top-3 classification with the supplied 129-class WikiArt checkpoint; verify `wikiart_target_id` against its label mapping.
- **Retain:** top-1 accuracy among 20 retain concepts and 80 COCO labels using CLIP ViT-L/14, higher is better.
- **CLIP:** mean paired image/text logit from CLIP ViT-B/32, higher is better.
- **FID:** Clean-FID against the matching real COCO image directory, lower is better.

CCE learns one token for 3,000 steps with independently seeded noise. UnlearnDiff optimizes a discrete prefix with straight-through token selection and probability-simplex projection: 5 tokens for nudity, 3 for dog/style, 40 updates at each of 50 timesteps. Both use DDIM sampling. Direct and retain sampling use PNDM; COCO uses DPM-Solver. All use 50 steps and guidance 7.5 by default.

Run selected splits or recompute metrics from completed generation:

```bash
python evaluate.py --config configs/dog.json --splits erased retain
python evaluate.py --config configs/dog.json --splits coco --metrics-only
```

Completed generations are reused only when their recorded configuration, checkpoint hash, and manifest hashes match. Incomplete generations are not accepted as finished experiments. Choose a new `evaluation.output` after interrupted generation or an attack. Changed generation settings require a new output directory.

## Slurm

Activate the environment and submit from the repository root. Resources belong to the submission command:

```bash
sbatch --gpus=1 --cpus-per-task=8 --mem=64G experiments/dog.sh retain
sbatch --gpus=1 --cpus-per-task=8 --mem=64G experiments/dog.sh train
sbatch --gpus=1 --cpus-per-task=8 --mem=64G experiments/dog.sh gallery
sbatch --gpus=1 --cpus-per-task=8 --mem=64G experiments/dog.sh evaluate
```

Wait for retain preparation before training, and for training plus gallery preparation before evaluation. Use the same task-specific overrides at every dependent stage.

## Tests

```bash
python -m unittest discover -s tests -v
sbatch --cpus-per-task=4 --mem=8G experiments/test.sh
```

Tests cover configuration validation, independent adaptive-momentum trajectories, optimizer restoration, guidance equations, manifest separation, trainable layer selection, complete small-model training with interruption and resume, frozen teacher parameters, CCE embedding isolation, CLIP embedding-path equivalence, and UnlearnDiff gradients. These checks do not establish full-scale experimental scores.
