<div align="center">

<h1> CrystaLLM-<span style="font-size: 1.2em;">&pi; </span> (property injection) </h1>
  <img src="images/Logo.png" alt="CrystaLLM-pi logo" width="150" />
  <p>
    <strong>A Transformer-based model for property-guided crystal structure generation
    </strong>
  </p>
</div>

<p align="center">
<a href="https://huggingface.co/c-bone">
    <img alt="Hugging Face" src="https://img.shields.io/badge/🤗%20Hugging%20Face-Models-blue.svg?style=plastic">
</a>
<a href="https://huggingface.co/spaces/LeMaterial/LeMat-GenBench">
    <img alt="Benchmark" src="https://img.shields.io/badge/🤗%20LeMat%20Bench-Unconditional%20Benchmark-lightblue.svg?style=plastic">
</a>
<a href="https://arxiv.org/pdf/2511.21299">
    <img alt="Preprint" src="https://img.shields.io/badge/Preprint-arXiv-red.svg?style=plastic">
</a>
<a href="https://www.nature.com/articles/s41467-024-54639-7">
    <img alt="Based on CrystaLLM Paper" src="https://img.shields.io/badge/Based%20on-CrystaLLM%20Paper-orange.svg?style=plastic">
</a>
<a href="https://github.com/C-Bone-UCL/CrystaLLM-pi-paper/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey.svg?style=plastic">
</a>

</p>

<br>

# Overview

CrystaLLM-<span style="font-size: 1.2em;">&pi;</span> is a Transformer-based system for generating crystalline structures as CIF files. It supports both unconditional generation and four conditional architectures that can generate structures based on target properties like bandgap, density, photovoltaic efficiency and XRD patterns.

<div align="center">
<img src="images/Framework_github.png" width="75%" style="background-color:white;"/>
</div>

## About this repository
> This repository reproduces the results of the CrystaLLM-&pi; paper,
> ["Discovery and recovery of crystalline materials with property-conditioned transformers"](https://arxiv.org/pdf/2511.21299).
> `main` reproduces the published (v2) results; branch
> [`paper_v1`](https://github.com/C-Bone-UCL/CrystaLLM-pi-paper/tree/paper_v1) preserves the pre-revision (v1) workflow.
> The maintained, up-to-date package lives at [C-Bone-UCL/CrystaLLM-pi](https://github.com/C-Bone-UCL/CrystaLLM-pi).


## Key Features

- **Unconditional Generation**: Generate crystal structures from structural/composition priors
- **Property-Guided Generation**: Generate crystal structures conditioned on target properties + structural priors
- **Multiple Architectures**: Choose from 4 different conditional methods plus unconditional base model
- **Flexible Conditioning**: You can use any set of numerical properties to condition, and one of the models handles heterogeneous datasets (some properties are missing in the dataset but not others...)
- **Evaluation of output structures**: Scripts for validity, uniqueness, novelty and stability metrics
- **HuggingFace Integration**: Pre-trained models available on HF Hub

## Table of Contents

- [Installation](#installation)
- [Model Types](#model-types)
- [LeMaterial Benchmark](#lematerial-benchmark)
- [Training, Generating & Evaluating from Scratch](#training-generating--evaluating-from-scratch)
- [Paper Studies](#paper-studies)
- [License](#license)
- [Contact](#contact)

<br>

# Installation

## Prerequisites

- Python 3.10+
- PyTorch 2.1+
- Conda for environment management
- Hugging Face and Weights & Biases accounts should be set up
- (Optional) CUDA-compatible GPU


## Setup

```bash
# Clone the repository
git clone https://github.com/C-Bone-UCL/CrystaLLM-pi.git
cd CrystaLLM-pi

# Create virtual environment
conda create -n CrystaLLM-pi_env python=3.10
conda activate CrystaLLM-pi_env

# Install dependencies and setup package
pip install -r requirements.txt
# material-hasher package needs to be installed via
pip install git+https://github.com/lematerial/material-hasher.git
# muon optimizer (addition in development, install required)
pip install git+https://github.com/KellerJordan/Muon
# Install CrystaLLM-pi in editable mode
pip install -e .
```

### Optional: ALIGNN Environment Setup

For property prediction (bandgap), set up a separate environment to avoid dependency conflicts:

```bash
conda create -n alignn_env python=3.10
conda activate alignn_env
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
pip install git+https://github.com/KellerJordan/Muon
pip install -r requirements-alignn.txt
```

### HuggingFace & Weights & Biases credentials

Create `API_keys.jsonc` in the root directory for HuggingFace and Weights & Biases integration (used by training, dataset upload, and prompt-making scripts):

```jsonc
// filepath: API_keys.jsonc
{
  "HF_key": "your_hf_key_here", // Hugging Face token
  "wandb_key": "your_wandb_api_key_here" // Weights & Biases key
}
```

<br>
<br>

# Model Types

CrystaLLM-<span style="font-size: 1.2em;">&pi;</span> supports one unconditional and four conditional model architectures, allowing for both standard and property-driven generation. The desired model can be selected during training using the `--activate_conditionality` flag.

> **Important:** In the paper, the `PKV` method is addressed as the `Prefix attention`, and `Slider` is called the `Residual attention`. For all intents and purposes, these are the exact same. However the codebase was developed with `PKV` and `Slider`, but their respective names were changed in the paper for technical clarity.

### 1. Unconditional CrystaLLM

Do not set the `--activate_conditionality` flag.

Standard CrystaLLM/GPT-2 architecture for generative tasks. Learns underlying patterns and grammar of CIF files without explicit property guidance.

### 2. Conditional Models

#### a. PKV-GPT (Prefix Attention)

`--activate_conditionality="PKV"`

Injects property information directly into the attention mechanism's past key-values. This allows the model to steer generation based on desired properties by concatenating conditional embeddings at each transformer layer. Provides strong conditioning while maintaining straightforward implementation. Based on ghost tokens from the [Prefix Tuning Paper](https://arxiv.org/abs/2101.00190).

<div align="center">
<img src="images/Prefix_github.png" width="75%" style="background-color:white;"/>
</div>

#### b. Slider-GPT (Residual Attention)

`--activate_conditionality="Slider"`

Novel architecture where conditioning information is dynamically injected into each attention block via a 'slider' mechanism. Features two separate attention mechanisms at every token generation: one for main text and one for conditions. Attention scores are combined via weighted sum. Handles missing or unspecified conditions with softer conditioning (weight initialized at 0 during finetuning).

<div align="center">
<img src="images/Residual_github.png" width="75%" style="background-color:white;"/>
</div>

<details>
<summary>Prepend and Raw model details (comparative baselines used in paper)</summary>

#### c. Prepend-GPT

`--activate_conditionality="Prepend"`

Prepends learned embeddings (soft prompts) to the input sequence. These prefix tokens represent desired conditional properties to guide model output. Provides strong conditioning with straightforward implementation but less flexibility than attention-based methods.

#### d. Raw-GPT

`--activate_conditionality="Raw"`

Baseline approach where numerical condition values are converted to text and appended to input prompts. Requires no architectural changes but increases sequence length. Implemented for comparison but generally less performant.

</details>

<br>
<br>

# LeMaterial Benchmark

CrystaLLM-<span style="font-size: 1.2em;">&pi;</span> was evaluated on [LeMaterial GenBench](https://huggingface.co/spaces/LeMaterial/LeMat-GenBench); the paper's LeMat-Benchmark appendix compares it against the other benchmarked generative models on MP-20 and Alex-MP-20 (headline numbers quoted in the Results section). The generation protocol used for the submissions is in [`A_Text_baseline.ipynb`](notebooks/A_Text_baseline.ipynb); scoring runs on the external LeMat-GenBench harness. (Leaderboard snapshot: top-5 on the default MSUN+SUN ranking, January 2026.)

### Key Takeaways

* **High-Fidelity Interpolation**: The model is in the top performers at generating structures that are closer to their relaxed equilibrium state than continuous generative models. (Relaxation RMSD)
* **Structural Diversity**: While the model replicates the training distribution with high precision (see distribution metrics), it maintains high element diversity - likely due to the large training set.
* **Tuning Exploration**: Lower novelty is a byproduct of high-fidelity distribution matching. To move away from the base distribution, users can increase the `temperature` parameter at inference or explore property conditioned options (like the SLME study)

<br>
<br>

# Training, Generating & Evaluating from Scratch

Complete pipeline for training your own models from data preprocessing to evaluation. All training and generation parameters and options are defined in [`_args.py`](_args.py). Training & generating should be done via configuration files (`.jsonc` format) which specify all necessary parameters.

> Maintained notebook workflow: [`notebooks/X_XRD_chili100k.ipynb`](notebooks/X_XRD_chili100k.ipynb) covers CHILI-100K preprocessing, second-pass Slider finetuning, conditioned generation, unconditional control runs, and aggregate metrics.

## Data Processing Pipeline

### Step 1: Data Preparation (**Required**)

Input data should be a pandas DataFrame saved as Parquet file. To train a model you should save a dataframe to a parquet file which contains:

**Required columns:**

* `Database`: Source database name
* `Reduced Formula`: Standard reduced chemical formula
* `CIF`: Crystallographic structure in CIF format

**For structure recovery benchmarks**

* `Material ID`: Database identifier required for structure recovery benchmarks

**Optional columns:**

* `<Property Columns>`: Target properties (e.g., "Bandgap (eV)", "Density (g/cm^3)")
* `condition_vector`: Pre-computed condition vectors (for XRD studies)

### Step 2: Deduplication and Filtering (Optional)

**Script:** `_utils/_preprocessing/_deduplicate.py` - Removes duplicate structures and filters invalid entries based on chemical formula and space group, keeping the structure with lowest volume per formula unit.

<details>
<summary>Example Usage and Args</summary>

```bash
python _utils/_preprocessing/_deduplicate.py \
  --input_file /path/to/raw_data.parquet \
  --output_parquet /path/to/deduplicated_data.parquet \
  --property_columns "['Bandgap (eV)', 'Density (g/cm^3)']" \
  --filter_na_columns "['Bandgap (eV)']" \
  --filter_zero_columns "['Density (g/cm^3)']" \
  --filter_negative_columns "['Bandgap (eV)']"
```

**Key arguments:**

* `--filter_na_columns`: Remove entries with N/A or NaN values
* `--filter_zero_columns`: Remove entries with zero values
* `--filter_negative_columns`: Remove entries with negative values

</details>

### Step 3: CIF Cleaning and Normalization (**Required**)

**Script:** `_utils/_preprocessing/_cleaning.py` - Standardizes CIF format and normalizes properties for stable training. Adds atomic property blocks, rounds numerical values, and applies variable brackets.

<details>
<summary>Example Usage and Args</summary>

```bash
python _utils/_preprocessing/_cleaning.py \
  --input_parquet /path/to/deduplicated_data.parquet \
  --output_parquet /path/to/cleaned_data.parquet \
  --num_workers 8 \
  --property_columns "['Bandgap (eV)', 'Density (g/cm^3)']" \
  --property1_normaliser "power_log" \
  --property2_normaliser "linear"
```

> Tip: Keep a note somewhere of the lowest and highest property values for each property, so that later when you have a particular property target you can easily normalize it to the format the model expects.

**Key arguments:**

* `--property1_normaliser` / `--property2_normaliser`: Normalization methods (`linear`, `power_log`, `signed_log`, `log10`, `None`)
* `--make_disordered_ordered`: Convert disordered structures to ordered ones
* `--num_workers`: Number of parallel workers for processing

**Normalization methods:**

* `linear`: Simple min-max scaling to [0,1] range
* `power_log`: Power transformation ($\beta$=0.8) followed by logarithmic scaling for skewed distributions
* `signed_log`: Signed logarithmic transformation for handling negative values
* `log10`: Base-10 logarithmic scaling for properties spanning multiple orders of magnitude
* `None`: No normalization applied

</details>

### Step 4: Dataset Upload to HuggingFace (**Required**)

**Script:** `_utils/_preprocessing/_save_dataset_to_HF.py` - Converts to HuggingFace format with train/validation/test splits and uploads to HF Hub.

> Important: You need to make sure that the data trained on has been passed through CIF cleaning, a quick way to make sure is check whether the CIFs in your dataframe contain brackets. If they do then text should be ready for training.

<details>
<summary>Example Usage and Args</summary>

```bash
python _utils/_preprocessing/_save_dataset_to_HF.py \
  --input_parquet /path/to/processed_data.parquet \
  --output_parquet "your-dataset-name" \
  --test_size 0.1 \
  --valid_size 0.1 \
  --HF_username "your-username" \
  --save_hub \
  --save_local
```

**Key arguments:**

* `--duplicates`: Prevents data leakage by splitting on Material ID (optional)
* `--test_size` / `--valid_size`: Split ratios (set both to 0.0 for training-only)
* `--save_hub` / `--save_local`: Upload to HF Hub and/or save locally (specify at least one)

</details>




## Training

All training should be done via configuration files (`.jsonc` format). These files specify model architecture, hyperparameters, data paths, and training settings. See example configs in `_config_files/training/` and review [`_args.py`](_args.py). for all available parameters.

> The `Muon` optimiser is now available for training, see [this blog post](https://kellerjordan.github.io/posts/muon/) for details. Importantly, you cannot use deepspeed when using muon. Simply do not feed a deepspeed configuration file and it will work fine (multi-GPU training still supported). Muon speeds up and stabilises training without any performance trade-offs (did some internal checks).

<details>
<summary>Base model training CLI example</summary>

### Base Model Pretraining

Train the unconditional base models from scratch:

```bash
python _train.py --config _config_files/training/unconditional/lematerial-small.jsonc
```

**Multi-GPU Training:**

```bash
torchrun --nproc_per_node=2 _train.py --config your_config.jsonc
```

</details>

<br>

<details>
<summary>Conditional finetuning CLI example</summary>

### Conditional Fine-tuning

Fine-tune pretrained base models for property-guided generation:

**Single GPU:**

```bash
python _train.py --config _config_files/training/conditional/ft-slme/slme_ft-PKV-opt.jsonc
```

**Multi-GPU:**

```bash
torchrun --nproc_per_node=2 _train.py --config _config_files/training/conditional/ft-slme/slme_ft-PKV-opt.jsonc
```

Loads pretrained weights as starting point (or trains from scratch), adds conditional architecture layers, and uses split optimizer with different learning rates for conditioning vs base layers.

</details>




## Advanced Generation Pipeline

### Step 1: Create Prompts

**Script:** `_utils/_generating/make_prompts.py` - Generate input prompts for conditional generation with different levels of structural information.

<details>
<summary>Examples of Prompt Construction and Args</summary>

**Manual Prompts:**

```bash
python _utils/_generating/make_prompts.py \
  --manual \
  --compositions "Na1Cl1,K2S1" \
  --condition_lists "0.2,0.0" "0.5,0.0" \
  --level "level_3" \
  --output_parquet "test_prompts.parquet"
```

**Automatic Prompts from Dataset:**

```bash
python _utils/_generating/make_prompts.py \
  --automatic \
  --HF_dataset "c-bone/mp_20_pxrd" \
  --split "test" \
  --level "level_2" \
  --condition_columns "Condition Vector" \
  --output_parquet "dataset_prompts.parquet"
```

**Prompt levels `--level`:**

* `level_1`: Minimal (unconditional generation)
* `level_2`: Composition only (default)
* `level_3`: Composition + atomic properties
* `level_4`: Up to space group information

**Composition-Condition Pairing modes `--mode`:**

Each quoted string is a **complete condition vector** (comma-separated property values).

* `cartesian` (default): All conditions applied to all compositions
* `paired`: 1:1 mapping - must have same count of conditions and compositions
* `broadcast`: Single condition applied to all compositions

</details>

### Step 2: Generate CIFs

**Script:** `_utils/_generating/generate_CIFs.py` - Generate crystal structures from prompts using trained models.

<details>
<summary>Examples of CIF generation and Args</summary>

```bash
python _utils/_generating/generate_CIFs.py \
  --config _config_files/generation/pkv_generation.jsonc
```

> You can generate with arguments from the CLI, but it's easier to use the config file. You can find a lot of examples in [`_config_files/generation`](_config_files/generation)

**Key generation settings:**

* **Temperature:** Controls randomness (default ~1.0, higher is more exploratory but higher chance of gibberish)
* **Top-p/Top-k:** Sampling parameters (typical: 0.95, 50)
* **scoring_mode:** if set to `None` and `target_valid_cifs = 0`, then we generate `max_return_attempts * num_return_sequences` CIFs per Prompt/Condition pair without validation. If set to `None` and `target_valid_cifs > 0`, then we validate generated CIFs and stop once that many valid CIFs are found, without ranking. If set to `LOGP`, we validate and rank using a perplexity based scoring method.
* **num_return_sequences:** Batch size for generation (adjust for GPU mem.)
* **max_return_attempts:** In raw mode, total generation for each Prompt/Condition pair = `max_return_attempts * num_return_sequences`. In validation-targeted modes, generation stops when `target_valid_cifs` valid CIFs are found or `max_return_attempts` is reached.

</details>

### Step 3: Post-process

**Script:** `_utils/_generating/postprocess.py` - Clean and validate generated CIF structures.

<details>
<summary>Examples of postprocessing and Args</summary>

```bash
python _utils/_generating/postprocess.py \
  --input_parquet "generated_cifs.parquet" \
  --output_parquet "processed_cifs.parquet" \
  --num_workers 4
```

Convcerts LLM outputs to standard Pymatgen style CIF format.

</details>




## Evaluation

### VUN Metrics (Validity, Uniqueness, Novelty)

**Script:** `_utils/_metrics/VUN_metrics.py` - Essential metrics for assessing generation quality using structural analysis.

**Required:** Structures must be post-processed with Reduced Formulas column included

**Metrics computed:**

* **Validity**: Structures with correct spacegroup, reasonable bond lengths, and consistent atom multiplicities
* **Uniqueness**: Distinct structures within the generated set (using BAWL hashing)
* **Novelty**: Structures not present in the reference dataset
* **Compositional Novelty**: Reduced Formula not present in reference dataset

<details>
<summary>Example Usage</summary>

```bash
python _utils/_metrics/VUN_metrics.py \
  --input_parquet generated_structures_processed.parquet \
  --huggingface_dataset "c-bone/mp_20" \
  --output_parquet vun_results.parquet \
  --num_workers 8
```

We can optionally set the `--check_comp_novelty` flag, which adds an `is_comp_novel` boolean column to the metrics dataframe.

</details>

### Energy Above Hull (Stability)

**Script:** `_utils/_metrics/mace_ehull.py` - Calculate thermodynamic stability using MACE energy predictions. See the [MACE paper](https://arxiv.org/abs/2206.07697) for details on the surrogate model.

> To calculate E_hull First, total energies are computed using the MACE-MP default calculator, predicted energies are then processed using the *MaterialsProject2020Compatibility* scheme to ensure consistency between GGA and GGA+U calculations. The surrogate energy predictions are compared to formation energies of known materials from the MP dataset and used to construct a convex hull. The energy above the convex hull (E_hull) quantifies thermodynamic stability by comparing a material’s formation energy to competing phases.

<details>
<summary>Example Usage and Args</summary>

```bash
python _utils/_metrics/mace_ehull.py \
  --post_parquet postprocessed_structures.parquet \
  --output_parquet stability_results.parquet \
  --num_workers 4
```

Lower E_hull values indicate higher thermodynamic stability. Structures with E_hull < 0.1 eV/atom are typically considered experimentally synthesizable. (We can extend to 0.157 eV/atom if we want to account for MAE in energy predictions of this MACE model)

</details>

### Additional Metrics

XRD, bandgap or density property metrics, VUN, and stability metrics are available in `_utils/_metrics/`.

> **Note**: ALIGNN-based scripts require the separate `alignn_env` environment.

# Paper Studies

The notebooks in [`notebooks/`](notebooks/) reproduce the studies in the paper end to end. Prefixes: `A_`/`B*` = baselines and property-conditioning studies, `X_` = full discovery/recovery pipelines, `Y_` = analyses and ablations. Figure/table numbers below follow the current arXiv version. Steps needing large external compute (pre-training, MatterGen training, DFT, LeMat-GenBench scoring) are marked inside the notebooks; the paper's framework schematics are hand-drawn and have no notebook source.

| Notebook | Study | Reproduces |
|---|---|---|
| [`A_Text_baseline.ipynb`](notebooks/A_Text_baseline.ipynb) | Unconditional text baselines (mp-20, alex-mp-20, LeMaterial) and LeMat-bench generation | Results headline numbers + LeMat-Benchmark appendix table (scoring via external LeMat-GenBench harness) |
| [`B1a_Pretrain_benefits.ipynb`](notebooks/B1a_Pretrain_benefits.ipynb) | Pretraining benefits for bandgap + E_hull conditional generation | Fig 3, Figs 13-14, Table VII |
| [`B1b_Mattergen.ipynb`](notebooks/B1b_Mattergen.ipynb) | Head-to-head comparison against MatterGen | Figs 15-17, MatterGen row of Table VII, Table IX provenance note |
| [`B2_Dataset_size_study.ipynb`](notebooks/B2_Dataset_size_study.ipynb) | Dataset-size study on density-conditioned generation | Fig 4, Fig 12, Table VIII |
| [`X_SLME.ipynb`](notebooks/X_SLME.ipynb) | Discovery pipeline for target photovoltaic efficiency (SLME) | Fig 5, Table I, Table XVI |
| [`X_XRD_chili100k.ipynb`](notebooks/X_XRD_chili100k.ipynb) | CHILI-100K XRD structure recovery | Fig 6 |
| [`X_XRD_jarvis.ipynb`](notebooks/X_XRD_jarvis.ipynb) | Jarvis-DFT XRD structure recovery | Table III, Table XII |
| [`X_XRD_mp-20.ipynb`](notebooks/X_XRD_mp-20.ipynb) | MP-20 theoretical-XRD structure recovery | Table II |
| [`X_XRD_TiO2.ipynb`](notebooks/X_XRD_TiO2.ipynb) | TiO2 polymorph recovery from experimental XRD (drives generation via [`_load_and_generate.py`](_load_and_generate.py)) | Table IV |
| [`Y_Dataset_stats.ipynb`](notebooks/Y_Dataset_stats.ipynb) | Dataset token/atom statistics | Table V, Fig 7 |
| [`Y_Logits.ipynb`](notebooks/Y_Logits.ipynb) | Digit-level logit analysis | Fig 9 |
| [`Y_Losses.ipynb`](notebooks/Y_Losses.ipynb) | Loss landscapes (paper appendices) | Figs 18-19 |
| [`Y_mp-20-xrd-ablations.ipynb`](notebooks/Y_mp-20-xrd-ablations.ipynb) | XRD-conditioning and perplexity-ranking ablations | Tables X-XI |

# Tokenizer

The `HF-cif-tokenizer` already contains everything you need to train/run models out of the box. However if for some reason a user wishes to add more tokens this can be done by:
- **Create the new vocab**: Edit the [`_create_vocab.py`](_utils/_tokenizer_utils/_create_vocab.py) file to include all the new tokens you want (if augmenting CIF with new tokens for example). Save a new `vocabulary.json` with the updated dictionary.
- **Optional: Add Spacegroups**: If new spacegroups are required for a particular study, these should be added to the [`spacegroups.txt`](_utils/_tokenizer_utils/spacegroups.txt) file.
- **Build New Tokenizer**: Once the new vocabulary is ready, just run the [`_save_tokenizer_to_HF.py`](_utils/_preprocessing/_save_tokenizer_to_HF.py) script, to save it locally or to HF. Then you can update the `pretrained_tokenizer_dir` argument in the train config to point to your new tokenizer!


# Citation

Please refer to the following when citing our work!
["Discovery and recovery of crystalline materials with property-conditioned transformers"](https://arxiv.org/pdf/2511.21299)

```
@misc{bone2025discoveryrecoverycrystallinematerials,
      title={Discovery and recovery of crystalline materials with property-conditioned transformers}, 
      author={Cyprien Bone and Matthew Walker and Kuangdai Leng and Luis M. Antunes and Ricardo Grau-Crespo and Amil Aligayev and Javier Dominguez and Keith T. Butler},
      year={2025},
      eprint={2511.21299},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2511.21299}, 
}
```

# License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

# Contact

For questions or support, please contact cyprien.bone.24@ucl.ac.uk or raise an issue on the GitHub page.

# Acknowledgments
This work has been supported by UKRI funding (EP/Y000552/1 and EP/Y014405/1)
