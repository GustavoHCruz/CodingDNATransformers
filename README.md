# CodingDNATransformers

This repository contains the code developed for my **master's thesis at UFPR**, which explores the application of **transformers** in the identification of **coding regions** in DNA sequences.

The project is entirely developed in Python and leverages state-of-the-art **transformer architectures** such as **GPT**, **BERT**, and **DNABERT**.

The contribution of this work is as fallows:

- Achieved **2nd place** at a national event in Fortaleza, Ceará, Brazil - [Symposium on Knowledge Discovery, Mining and Learning (KDMiLe) - SBC](https://doi.org/10.5753/kdmile.2025.247575).
- Later accepted for publication in Athens, Greece, on [International Conference on BioInformatics and BioEngineering (BIBE) - IEEE]().

---

## Approaches

This work is structured into three main approaches, each addressing the problem of DNA coding analysis from a different perspective.  
Each section describes the models, expected input/output format, and examples.

---

### 1. Full DNA Sequence Classification (Exon/Intron)

The model receives an entire DNA sequence and classifies it as **coding (exon)** or **non-coding (intron)**.  
Additional metadata (organism, gene name, flanking regions) can be included probabilistically to enrich the context.

- **Models used:** [GPT-2](https://huggingface.co/openai-community/gpt2), [BERT](https://huggingface.co/google-bert/bert-base-uncased), [DNABERT2](https://huggingface.co/zhihan1996/DNABERT-2-117M)

#### Input/Output Format (GPT and BERT)

```text
<|SEQUENCE|>ACGAAGGGTAAGCC...
<|FLANK_BEFORE|>ACGT...
<|FLANK_AFTER|>ACGT...
<|ORGANISM|>homo sapiens
<|GENE|>...
<|TARGET|>
```

- `<|SEQUENCE|>`: the full DNA sequence.
- `<|FLANK_BEFORE|>` and `<|FLANK_AFTER|>`: optional context regions.
- `<|ORGANISM|>`: optional organism name (truncated).
- `<|GENE|>`: optional gene name (truncated).
- `<|TARGET|>`: separation token where the model predicts the label.

### 2. Sliding Window Nucleotide Classification (I/E/U)

Uses a sliding window of 1 nucleotide to classify each position in a DNA sequence as Intron (I), Exon (E), or Unknown/Uncertain (U).
This enables fine-grained annotation across long DNA sequences.

- Models used: [BERT](https://huggingface.co/google-bert/bert-base-uncased)

#### Input/Output Format

```text
<|SEQUENCE|>A
<|FLANK_BEFORE|>CGAA...
<|FLANK_AFTER|>GGTA...
<|ORGANISM|>homo sapiens
<|TARGET|>
```

- `<|SEQUENCE|>`: the current nucleotide.
- `<|FLANK_BEFORE|>` and `<|FLANK_AFTER|>`: local nucleotide context.
- `<|ORGANISM|>`: optional organism name.
- `<|TARGET|>`: separation token where the label is predicted.

### 3. Direct DNA-to-Protein Translation

The model translates DNA sequences directly into their corresponding protein sequences, predicting amino acids from codons.

Organism metadata can optionally be provided.

- Models used: [GPT-2](https://huggingface.co/openai-community/gpt2)

#### Input/Output Format

```text
<|DNA|>ATGAAATTT...
<|ORGANISM|>homo sapiens
<|PROTEIN|>
```

- `<|DNA|>`: the input DNA sequence.
- `<|ORGANISM|>`: optional organism name.
- `<|PROTEIN|>`: token where the amino acid sequence is generated.

## Configuration and Execution

Each approach has its own main notebook file:

- Full Sequence Classification: main-exin.ipynb
- Sliding Window I/E/U: main-nucl.ipynb
- DNA to Protein Translation: main-dna.ipynb

The code can be executed on both CPU and GPU, with CUDA strongly recommended for optimal performance.

## Model Implementation and Usage

All implemented models are wrapped into Python classes that abstract their functionality.

This ensures a clean interface and allows users to easily adapt the models to their own workflows.

- `.build_input`

  Defines how inputs are structured for each model, mapping the required fields (DNA, organism, flanking regions, etc.) to the proper format.

- `.generate`

  Produces predictions according to the model type:

  - For sliding window models, performs position-wise inference.
  - For classification models, outputs a single label (e.g., exon or intron).
  - For generative models, produces the next token(s) (e.g., amino acids for DNA to protein).

- `.from_pretrained (custom implementation)`

  Loads models directly from a Hugging Face repository (future links to be provided).

All models can be used independently of the provided pipelines.

The notebooks are offered as optional pipelines for training, inference, and evaluation, but users can directly import and use the model classes in their own code (links to HuggingFace below).

## Datasets

[GenBank](https://www.ncbi.nlm.nih.gov/genbank/) dataset was used on this work.

The processed version used in this work is available on the [here](link_here).

The derivations from it, used to train the models, was also provided in the following links:

- [Full DNA Sequence Classification Train Dataset](link_here)
- [Sliding Window Nucleotide CLassification Dataset](link_here)
- [Direct DNA-to-Protein Translation Dataset](link_here)

## Models

Each one of the trained models will be published on Hugging Face:

- **Full DNA Sequence Classification:**
  - [GPT-2](link_here)
  - [BERT](link_here)
  - [DNABERT2](link_here)
- **Sliding Window Nucleotide CLassification:**
  - [BERT](link_here)
- **Direct DNA-to-Protein Translation Dataset:**
  - [GPT-2](link_here)

## Optional Dependency: BLASTp

The evaluation results presented in the articles for the DNA-to-Protein model were generated using **BLASTp** as an additional validation step.

This tool is **not required** for running the models or using the pipeline itself, but it is a **dependency** if one wishes to reproduce the evaluation process provided in the accompanying notebooks.

**BLASTp is only necessary to run the evaluation notebooks that compare predicted protein sequences against reference databases**

## Articles and Commit History

The repository also contains earlier versions of the project that correspond to the manuscripts submitted to scientific conferences.

These versions are preserved in past commits, providing a reproducible record of how the methods and experiments evolved over time.

This ensures full transparency and traceability of the results published in the Fortaleza and Athens papers.
