# rTGN-SVDD

Companion code for
[Noise Robust One-Class Intrusion Detection on Dynamic Graphs](https://arxiv.org/abs/2508.14192).
The repository implements the TGN-SVDD baseline and its probabilistic,
noise-robust extension rTGN-SVDD, with configuration-driven training and
inference.

## Installation

Python 3.10--3.12 is supported. Create either the CPU environment:

```bash
PYTHONNOUSERSITE=1 conda env create -f environment.yml
conda activate rtgn_svdd
```

or the CUDA environment:

```bash
PYTHONNOUSERSITE=1 conda env create -f environment-cuda.yml
conda activate rtgn_svdd_cuda
```

The package can also be installed in an existing compatible environment:

```bash
python -m pip install -e ".[plotting]"
```

## Data

Process CIC-IDS2017 using the public
[TGN-SVDD preprocessing pipeline](https://github.com/AlekseiLiu/tgn_svdd), in
particular
[`data_processing/cic_2017_preprocess.py`](https://github.com/AlekseiLiu/tgn_svdd/blob/main/data_processing/cic_2017_preprocess.py).

Place the generated Monday-plus-day CSVs at:

```text
data/wednesday.csv
data/tuesday.csv
```

Alternatively, update the paths in `configs/demo.json`. The files must use a
consistent node-ID mapping and feature order.

## Demo

Review `configs/demo.json`, select `cpu` or `cuda:0`, and adjust the data paths
if necessary. Validate the configuration with:

```bash
rtgn-svdd validate --config configs/demo.json
```

Run the example training and inference workflow with:

```bash
bash scripts/run_demo.sh configs/demo.json
```

Generated checkpoints, scores, metrics, and plots are written below the
configured `experiment.output_root`.

## Method

Both models combine TGN memory, temporal-neighbor sampling, and graph-attention
embeddings with a one-class SVDD objective. TGN-SVDD scores events by their
squared distance from a learned center. rTGN-SVDD additionally models a
diagonal Gaussian and provides predictive variance as an uncertainty score.

## Citation

```bibtex
@article{liuliakov2024noise,
  title={Noise Robust One-Class Intrusion Detection on Dynamic Graphs},
  author={Liuliakov, Aleksei and Schulz, Alexander and Hermes, Luca and Hammer, Barbara},
  journal={ESANN 2024 proceedings},
  year={2024}
}
```

## License

This project is available under the [MIT License](LICENSE).
