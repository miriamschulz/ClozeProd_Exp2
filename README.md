# About

This repository contains backups of the files for the second Cloze Prediction & Production SPR experiment.

# Folders and files

## `stimuli`

Directory containing the scripts for stimuli selection, checks and preprocessing.

### a. `stone_2023`

Folder containing the items used by Stone et al. 2023 (based on Nicenboim et al. 2020's condition a), and scripts to annotate them for:

- target word length
- target word frequency (German SUBTLEX)
- fasttext embedding similarity of the target noun pair for each item
- LLM surprisal for each target (GerPT2, GerPT2-large, Leo-7B, Leo-13B)

However, due to issues with zero cloze condition b targets and some annotation erros with this item set, the items could not be adapted as such; instead, the original Nicenboim et al. 2020 Cloze data were inspected to adapt some items.

### b. `nicenboim_2020_cloze`

Folder containing the scripts to re-estimate all target cloze probabilities from the original raw Nicenboim et al. 2020 cloze data (for condition a: Constraining-a), and to change target nouns for the Stone et al. 2023 stimuli where necessary, or discard items that could not be fixed.

### c. `chosen`

Folder containing the scripts to choose and inspect the items from the Stone et al. 2023/Nicenboim et al. 2020 data set that match the defined criteria.


## `power_analysis`

Directory containing the power analysis [TODO].

## `pcibex`

Directory containing the code for the PCIbex implementation of the experiment [TODO].

## `results`

Directory containing the analysis scripts (preprocessing, analysis, plots)  [TODO].
