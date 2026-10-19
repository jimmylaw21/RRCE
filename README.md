RRCE
====

Standard RRCE for nudity, dog, and Van Gogh erasure with Stable Diffusion v1.4.

Python 3.10+, PyTorch with CUDA, and a GPU with sufficient memory are required.

```bash
pip install -r requirements.txt
python -m experiments.prepare --i2p data/i2p.csv --coco data/coco30k.csv
sbatch --gpus=1 --cpus-per-task=8 --mem=64G experiments/dog.sh --stage train
sbatch --gpus=1 --cpus-per-task=8 --mem=64G experiments/dog.sh --stage evaluate
```

Use `nudity.sh` or `van_gogh.sh` for the other tasks. Activate the Python environment before submission. Training must finish before evaluation. Optional paths: `--model`, `--data-root`, and `--output-root`.

Inputs:

- `data/{nudity,dog,van_gogh}/target/`: training target images, separate from evaluation images.
- `data/i2p.csv`: I2P prompts with `prompt,nudity_percentage` columns.
- `data/coco30k.csv`: fixed COCO-30K rows with `case_number,prompt,evaluation_seed,coco_id` columns.
- `data/models/nudenet.onnx`: NudeNet detector used for nudity evaluation.
- `data/models/wikiart/`: 129-class WikiArt ViT classifier and image processor, with the Van Gogh class named in `id2label`.
- `data/coco_reference/`: the corresponding 30,000 real COCO images.

Training uses three defense blocks of 500 ordered erase/retain pairs, two 3,000-step recovery rounds, learning rate `2e-5`, adaptive first momentum in `[0.1,0.9]`, fixed second momentum `0.999`, and a frozen pretrained teacher. Dog and nudity update the full U-Net; Van Gogh updates cross-attention.

Evaluation writes Erased, UD, CCE, Retain, CLIP, and FID to `outputs/<task>/results.json`. CCE uses 30 dog images, 6 Van Gogh images, or a disjoint NudeNet-filtered I2P gallery. UD performs 40 updates at each of 50 diffusion timesteps with 5 nudity tokens or 3 dog/style tokens. Image generation uses 50 steps and guidance `7.5`.

Datasets, generated images, model weights, and experiment results are external inputs or outputs.
