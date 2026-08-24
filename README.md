# Assessing AI Scoring Accuracy for De Novo Proteins

This repository contains the code and analysis for my Master's thesis, conducted at the **University of Copenhagen (UCPH)** in the **Data Science for Drug Design (DSDD)** group.

The project investigates the ability of AI-based biomolecular structure prediction models to identify experimentally validated de novo protein binders targeting **EGFR**.

## Project Overview

The study uses a dataset of **400 experimentally validated de novo proteins targeting EGFR**, provided by **AdaptyvBio**.

The dataset is evaluated using predictions from multiple AI-based structure prediction models. The aim is to investigate the predictive performance of individual model outputs and determine whether **agreement between models** can provide additional information for identifying successful protein binders.

The project focuses on:

* Evaluating confidence and scoring metrics across AI models
* Comparing structural predictions between models
* Identifying correlations between different model outputs
* Quantifying structural and interface agreement between models
* Assessing whether consensus across models improves binder identification

## AI Models

The analysis includes the following structure prediction models:

* AlphaFold2
* AlphaFold3
* Boltz-2
* Chai-1
* HelixFold3
* OpenFold2
* OpenFold3
* Protenix
* SeedFold

## Analysis

The analysis combines **model-native confidence metrics**, **structural comparison metrics**, and **statistical and machine learning approaches**.

### Native Metrics

* pLDDT
* iPTM
* Ranking score

### Structural Agreement

* Binder-aligned RMSD
* Pocket-aligned RMSD
* Interface similarity using the Jaccard index

### Statistical Analysis

* Mann–Whitney U tests
* Spearman correlation
* ROC AUC analysis

### Machine Learning

* Symbolic machine learning using QLattice

## Workflow

```text
Experimental dataset
        │
        ▼
AI-based structure predictions
        │
        ▼
Metric extraction
        │
        ├── Native confidence metrics
        │
        └── Structural/interface metrics
        │
        ▼
Cross-model comparison
        │
        ▼
Statistical analysis
        │
        ▼
Predictive performance evaluation
        │
        ▼
Consensus / combined scoring
```

## Repository Structure

All the source code for the analysis is found within the /Analysis directory.

## Thesis

**Title:**
*Evaluating AI Complex Prediction Methods Through Consensus and Agreement Analysis of De Novo Protein Binders Targeting EGFR*

**Institution:** University of Copenhagen (UCPH)
**Research Group:** Data Science for Drug Design (DSDD)

## Acknowledgements

The dataset used in this project was provided by **AdaptyvBio**.

The project was conducted as part of the Data Science for Drug Design group at the University of Copenhagen.
