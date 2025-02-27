# Splicing Sites Identification in DNA Using Transformers

This repository contains the code developed for my **master's thesis**, which explores the application of **transformers** in the identification of **coding regions** in DNA sequences. The project is entirely developed in Python and includes different approaches to classify and predict intron and exon regions using **deep learning models**.

## Approaches

This work is structured into three main approaches, each with its own execution flow:

1. Intron and Exon Sequence Classification (ExInSeqs)
- The model receives DNA segments and classifies each as intron or exon.
- Models used:
  - GPT2 (normal, medium, large, xl)
  - bert-base-uncased
  - DNABERT
- Example:
```
prompt: AGGGTA...
target: INTRON
```

2. Coding Region Prediction in Full Sequences (RebuildSeqs)
- The model processes a complete DNA sequence and identifies regions, explicitly marking them.
- Models used:
  - GPT2 (normal, medium, large, xl)
- Example:
```
prompt: ACGAAGGGTAAGCC...
target: [EXON]ACGA[EXON][INTRON]AGGGTA[INTRON][EXON]AGCC...
```

3. Sliding Window with Flanking Context (SWExIn)
- Uses a sliding window to classify whether a trinucleotide belongs to an intron or exon, considering previous and next context.
- Model used:
  - bert-base-uncased
- Example:
```
ACGAAGGGTAAGCC...
EEEEIIIIIIEEEE...
```

## Configuration and Execution

Each approach has its own main script (and Jupyter):
- ExInSeqs: main_ExInSeqs.py
- RebuildSeqs: main_RebuildSeqs.py
- SWExIn: main_SWExIn.py

All approaches share a single configuration file:

```
configs/config.json
```

This file contains hyperparameters and model settings, which can be adjusted as needed.

## Recommended Execution

- The code can be executed on both CPU and GPU, but for optimal performance, CUDA is recommended.
- To ensure reproducibility, set the following environment variable before execution:

```
export TF_ENABLE_ONEDNN_OPTS=0  # Linux/macOS
set TF_ENABLE_ONEDNN_OPTS=0     # Windows (CMD)
$env:TF_ENABLE_ONEDNN_OPTS=0    # Windows (PowerShell)
```

## Dataset

- Scripts for dataset creation for each experiment are provided.
- To ensure comparable results, the original dataset is recommended.

For any questions or contributions, feel free to open an issue or submit a pull request.