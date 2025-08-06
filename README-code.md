# Core Module

The `core` module contains the fundamental components of the BFF (Borrowing From the Future) framework, including:

## Key Components

### Models
- `models.py`: Implements various embedding models including:
  - `CBOW`: Continuous Bag of Words model
  - `GloVe`: Global Vectors for word representation
  - `BERT`: Bidirectional Encoder Representations from Transformers
  - `PositionalEncoder`: For positional embeddings in transformer models

### Data Utilities
- `data_utils.py`: Contains data processing classes and utilities:
  - `Corpus`, `GloVeDataset`, `BERTPretainingData` dataset classes
  - `BERTCollate` for batch processing
  - `MultiModalDataGenerator` and `MultiModalDataset` for multi-modal data handling

### Training Components
- `trainer.py`: Training framework with:
  - `Trainer` base class
  - Specialized trainers like `BERTTrainer`
  - `EarlyStopping` utility

### Downstream Models
- `downstream_models.py`: Contains models for specific tasks:
  - `RNNEncoder` for sequence processing
  - `ForecastingAE` for autoencoding
  - Various mixing modules (`SelfAttnMixer`, `SelfGatingMixer`)
  - `BackboneModel` as the main model architecture

### Loss Functions
- `losses.py`: Implements custom loss functions including:
  - `pairwise_cosine` for similarity calculation
  - `snn_loss` for supervised contrastive learning

### Bootstrap Utilities
- `bootstrap.py`: Contains bootstrapping classes for evaluation:
  - `AUCBootstrapping` for standard classification
  - `CDAUCBootstrapping` for survival analysis

## Utils Submodule

The `utils/` directory contains utility functions:
- `bert_utils.py`: BERT-specific utilities
- `tokenizer_utils.py`: Tokenizer constants and special tokens

## Usage

This module serves as the core implementation of the BFF framework, providing:
1. Fundamental neural network architectures
2. Data processing pipelines
3. Training and evaluation utilities
4. Specialized components for multi-modal learning and temporal modeling

For specific usage examples, refer to the training scripts in the parent directory.