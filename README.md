# CodingDNATransformers

This repository contains the code developed for my **master's thesis at UFPR**, which explores the application of **transformers** in the identification of **coding regions** in DNA sequences.

The project is entirely developed in Python and leverages state-of-the-art **transformer architectures** such as **GPT**, **BERT**, and **DNABERT**.

The contribution of this work is as fallows:

- **Full Paper – 2nd Place (National)**  
  Achieved **2nd place** at the _Symposium on Knowledge Discovery, Mining and Learning (KDMiLe 2025)_, organized by the Brazilian Computer Society (SBC), held in Fortaleza, Ceará, Brazil.  
  [https://doi.org/10.5753/kdmile.2025.247575](https://doi.org/10.5753/kdmile.2025.247575)
- **Short Paper (International)**  
  Presented at the _IEEE International Conference on Bioinformatics and BioEngineering (BIBE 2025)_, held in Athens, Greece.  
  [https://doi.org/10.1109/BIBE66822.2025.00113](https://doi.org/10.1109/BIBE66822.2025.00113)

---

## Tasks

This work is structured into three main tasks, each addressing the problem of DNA coding analysis from a different perspective.  
Each following section describes the models, expected input/output format, and examples.

---

### 1. Full DNA Sequence Classification (Exon/Intron)

The model receives an entire DNA sequence and classifies it as **coding (exon)** or **non-coding (intron)**.  
Additional metadata (organism, gene name, flanking regions) can be included probabilistically to enrich the context.

- **Models used:** [GPT-2](https://huggingface.co/openai-community/gpt2), [BERT](https://huggingface.co/google-bert/bert-base-uncased), [DNABERT-2](https://huggingface.co/zhihan1996/DNABERT-2-117M), [T5](https://huggingface.co/google-t5/t5-base)

#### Input/Output Format (GPT and BERT)

```text
<|SEQUENCE|>[A][C][G][A][A][G][G][G][T][A][A][G][C][C]...
<|FLANK_BEFORE|>[A][C][G][T]...
<|FLANK_AFTER|>[A][C][G][T]...
<|ORGANISM|>homo sapiens
<|GENE|>...
<|TARGET|>
```

- `<|SEQUENCE|>`: token for the full DNA sequence.
- `<|FLANK_BEFORE|>` and `<|FLANK_AFTER|>`: optional context regions.
- `<|ORGANISM|>`: optional organism name (truncate to a maximum of 10 characters).
- `<|GENE|>`: optional gene name (truncate to a maximum of 10 characters).
- `<|TARGET|>`: separation token where the model predicts the label.

### 2. Sliding Window Nucleotide Classification (I/E/U)

Uses a sliding window of 1 nucleotide to classify each position in a DNA sequence as Intron (I), Exon (E), or Unknown/Uncertain (U).
This enables fine-grained annotation across long DNA sequences.

- Models used: [BERT](https://huggingface.co/google-bert/bert-base-uncased)

#### Input/Output Format

```text
<|SEQUENCE|>[A]
<|FLANK_BEFORE|>[C][G][A][A]...
<|FLANK_AFTER|>[G][G][T][A]...
<|ORGANISM|>homo sapiens
<|TARGET|>
```

- `<|SEQUENCE|>`: token for the current nucleotide.
- `<|FLANK_BEFORE|>` and `<|FLANK_AFTER|>`: local nucleotide context.
- `<|ORGANISM|>`: optional organism name.
- `<|TARGET|>`: separation token where the label is predicted.

### 3. Direct DNA-to-Protein Translation

The model translates DNA sequences directly into their corresponding protein sequences, predicting amino acids from codons.

Organism metadata can optionally be provided.

- Models used: [GPT-2](https://huggingface.co/openai-community/gpt2), [T5](https://huggingface.co/google-t5/t5-base)

#### Input/Output Format

```text
<|DNA|>[DNA_A][DNA_T][DNA_G][DNA_A][DNA_A][DNA_A][DNA_T][DNA_T][DNA_T]...
<|ORGANISM|>homo sapiens
<|PROTEIN|>[PROT_C][PROT_A][PROT_G]...
```

- `<|DNA|>`: start of the input of the DNA.
- `<|ORGANISM|>`: optional organism name.
- `<|PROTEIN|>`: start of the amino acid sequence output.

## Configuration and Execution

Each approach has its own main notebook file:

- Full Sequence Classification: main-exin.ipynb
- Sliding Window I/E/U: main-nucl.ipynb
- DNA to Protein Translation: main-trad.ipynb

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

The processed version used in this work is available on [DNA Coding Regions Dataset](https://huggingface.co/datasets/GustavoHCruz/DNA_coding_regions).

## Models

Some of the trained models are published on Hugging Face:

- **Full DNA Sequence Classification:**
  - [GPT-2](https://huggingface.co/GustavoHCruz/ExInGPT)
  - [BERT](https://huggingface.co/GustavoHCruz/ExInBERT)
  - [DNABERT2](https://huggingface.co/GustavoHCruz/ExInDNABERT2)
- **Sliding Window Nucleotide Classification:**
  - [BERT](https://huggingface.co/GustavoHCruz/NuclBERT)
  - [DNABERT2](https://huggingface.co/GustavoHCruz/NuclDNABERT2)

## Optional Dependency: BLASTp

The evaluation results presented in the articles for the DNA-to-Protein model were generated using **BLASTp** as an additional validation step.

This tool is **not required** for running the models or using the pipeline itself, but it is a **dependency** if one wishes to reproduce the evaluation process provided in the accompanying notebooks.

**BLASTp is only necessary to run the evaluation notebooks that compare predicted protein sequences against reference databases**

## Articles and Commit History

The repository also contains earlier versions of the project that correspond to the manuscripts submitted to scientific conferences.

These versions are preserved in past commits, providing a reproducible record of how the methods and experiments evolved over time.

This ensures full transparency and traceability of the results published in the Fortaleza and Athens papers.
