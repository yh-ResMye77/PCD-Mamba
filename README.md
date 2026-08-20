# PCD-Mamba

Official code scaffold for **Phase-Conditioned Decoupled Mamba for Underwater Image Enhancement with Noise-Resistant Structural Priors**.


## Environment

```bash
pip install -r requirements.txt
```

This implementation uses the Mamba-1 block API, `mamba_ssm.Mamba`.
Install `mamba-ssm` and `causal-conv1d` versions that match your CUDA and PyTorch environment.

## Configuration

Default paper settings are recorded in:

```text
configs/pcd_mamba_default.yaml
```

The training and evaluation scripts use command-line arguments for paths and runtime settings. The YAML file documents the released model, data, and optimization defaults for reproducibility.

## Dataset Layout

The UIEB public benchmark can be downloaded from:

```text
https://li-chongyi.github.io/proj_benchmark.html
```

After downloading and splitting the paired images, arrange the dataset as follows:

```text
UIEB/
  train/
    hazy/
      *.png
    clear/
      *.png
  val/
    hazy/
      *.png
    clear/
      *.png
```

The scripts also accept `input/groundtruth` as folder names:

```text
dataset_root/
  train/
    hazy/
    clear/
  val/
    hazy/
    clear/
```

File names are matched by exact file name first, then by stem.


## Train

```bash
python train_pcd_mamba.py \
  --private_dir /path/to/dataset_root \
  --results_dir results/PCD_Mamba \
  --train_batch_size 8 \
  --total_epochs 600 \
  --gpu 0
```

## Evaluate

```bash
python eval_pcd_mamba.py \
  --private_dir /path/to/dataset_root \
  --split test \
  --checkpoint results/PCD_Mamba/models/best_psnr.pth \
  --results_dir results/PCD_Mamba_eval \
  --gpu 0
```

## Citation

Please cite the paper after the final publication metadata is available.
