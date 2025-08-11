# AMPPred - Antimicrobial Peptide Prediction Tool

AMPPred is a Python package for predicting antimicrobial peptides (AMPs) from nucleotide sequences using pre-trained BERT-based models. The tool can work with both raw nucleotide sequences and coding sequences extracted using gene prediction algorithms.

## Training Data & Performance

The model has been trained on high-quality, curated datasets from established antimicrobial peptide databases:

- **Positive Samples (AMPs)**: [APS Database (UNMC)](https://aps.unmc.edu/) & [dbAMP (CUHK)](https://awi.cuhk.edu.cn/dbAMP/)
- **Negative Samples (Non-AMPs)**: NCBI protein sequences, cross-verified and validated
- **Test Accuracy**: **99.05%** (0.990473)

The training dataset combines comprehensive antimicrobial peptide collections with carefully selected and validated non-AMP sequences, ensuring robust classification performance across diverse sequence types.

## Features

- **AMP Prediction**: Classify sequences as antimicrobial peptides or non-AMPs
- **Gene Prediction**: Extract coding sequences from nucleotide sequences using Pyrodigal
- **Model Management**: Download and manage pre-trained models from Hugging Face
- **Flexible Input**: Support for FASTA format nucleotide sequences
- **Batch Processing**: Efficient processing of multiple sequences
- **Results Export**: Generate CSV reports and optional FASTA files of predicted AMPs

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch
- CUDA (optional, for GPU acceleration)

### Install from source

```bash
git clone https://github.com/neeraj1014/Antimicrobial-Peptide-Predictor.git
cd amppred
pip install -e .
```

### Dependencies

The tool automatically installs the following dependencies:

- `torch` - PyTorch for deep learning
- `transformers` - Hugging Face transformers library
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `pyrodigal` - Gene prediction
- `biopython` - Biological sequence analysis
- `huggingface_hub` - Model downloading and management

## Quick Start

### 1. Download a pre-trained model

```bash
# Download the default model
amp-build --model Neeraj0101/AMP-Predict

# List available models
amp-build --list_models
```

### 2. Run predictions

```bash
# Basic usage with nucleotide sequences
amp-run --input sequences.fasta

# Use coding sequences directly (skip gene prediction)
amp-run --input sequences.fasta --codon

# Generate FASTA file of predicted AMPs
amp-run --input sequences.fasta --get_amp

# Use a specific model
amp-run --input sequences.fasta --model your-model-name
```

## Usage

### Command Line Interface

#### Model Management (`amp-build`)

Download and manage pre-trained models:

```bash
# Download default model
amp-build --model Neeraj0101/AMP-Predict

# Download custom model
amp-build --model huggingface_model [https://huggingface.co/Neeraj0101]

# List downloaded models
amp-build --list_models
```

#### Prediction (`amp-run`)

Run AMP predictions on your sequences:

```bash
amp-run --input <fasta_file> [OPTIONS]
```

**Options:**
- `--input`: Path to input FASTA file (required)
- `--codon`: Use sequences directly without gene prediction
- `--model`: Specify model name (default: `Neeraj0101/AMP-Predict`)
- `--get_amp`: Generate FASTA file containing only predicted AMPs

### Input Format

AMPPred accepts nucleotide sequences in FASTA format:

```
>sequence_1
ATGAAATACCTGTTGCCGACCGCACAACGCCTGTACGAGATCAAGAAGGTAAAACTGGTAATCTGGATTAACGGCGATAAAGGCTATAACGGTCTCGCTGAAGTCGGTAAGAAATTCGAGAAAGATACCGGAATTAAAGTCACCGTTGAGCATCCGGATAAACTGGAAGAGAAATTCCCACAGGTTGCGGCAACTGGCGATGGCCCTGACATTATCTTCTGGGCACACGACCGCTTTGGTGGCTACGCTCAATCTGGCCTGTTGCTGAACATCACCCAACGCCTGTTGCCCGACCCAGGTCCACAGCTGGAAAGCTTCCTGCAAAAGTACGGCCCCGTTCCGGCAGCAACCAAGACCACCGGTTGGTGTTGCCGACACCGCACAAACCCAGCGCCAGACGCCTGGCTTATCGGGAAACCCCCGGACGGCCCCTACATGGTTGATGTGGACTCTACCGGCTCAAATCTGCCGGAAGACATTCGCAAAATTGATAAACAGGGTATGGTTGTCGACCGGCAGAGTTCCGGAGAGAAACCTGACTGGCGAAGGCGAAAACGACCTAGGTGTCCCCGGCGGTCCCTGGGGGACGGTGATGGCCACTATCACCGGCTATGTGAGACTCGGACTTGAACTAGTGTACTTCCCTAGACCTTTCGGTGTGATGGAACAAGCCGTACGCAACGTGCTTAACCTGACCAAAAACAATGGAGAGCGAATCGTGCGCGTTATCGATGAAAACATCCTATCCACCCCACATCCCGCCGCACAGGAGTTGTTCGCCGTGCCCACCACGCCTGCGCAGAACCCACCCATTATTGGTGCACTTGCGAAGGCAAATGGACAGGATGATGAAGGGCGCATGGGCGCAATTAAAGAAGATGGCATCCTCCACCTATGGCTAGTGATCGCCAACCCTGTGCTGGCGATTGCGAAAGGGTAG

>sequence_2
ATGGCTAAGTACCTGTTACCGACTGCGCAACGTCTGTACGAAATCAAGAAAGTGAAACTGGTGATCTGGATTAACGGCGATAAAGGATATAACGGCCTGGCTGAGGTGGGTAAAAGATTCGAAAAGGACACCGGAATCAAAGTGACCGTGGAACACCCGGACAAACTGGAGGAGAAATTCCCGCAGGTTGCGGCTACCGGAGACGGTCCCGACATTATCTTCTGGGCACACGACCGCTTCGGCGGCTACGCTCAGTCTGGCCTGTTGCTGAACATCACCGAACGCCTGTTGCCCGACCCGGGGCCACAACTGGAATCCTTTCTGCAGAAGTACGGACCGGTGCCGGCGGCCACCAAGACCACCGGGTTGGTGTTGCCGACACCGCTCAAACCCAACGCCAGACGCCTGGCCTACCGCGAGACCCCGGGACGGCCGCTACATGGTGGACGTGGACAGCACCGGCTCTAATCTGCCGGAAGATATTCGCAAAATTGACAAGCAGGGTTTGGTGATAGACCGGCAGAGCTCCGGAGAAAACCTGACCGGCGAAGGCGAAAACGATCTGGGCGTGCCGGGTGGCCCGTGGGGAACCGTGATGGCCACTATCACCGGCTACGTGCGACTGGGCCTTGAACTAGTGTACTTCCCCCGCCCTTTCGGTGTGATGGAACAAGCCGTGCGCAACGTCCTTAACCTGACCAAAAACAACGGTGAACGCATCGTGCGCGTGATCGACGAAAACATCCTGTCCACCCCCCACCCGGCCGCTCAAGAGTTGTTCGCGGTTCCGACCACGCCCGCTCAGAACCCCCCGATTATTGGGGCCCTTGCGAAGGCAAACGGCCAGGATGATGAAGGACGTATGGGCGCCATTAAGGAGGACGGCATCCTCCACCTATGGCTCGTGATCGCCAACCCGGTGCTGGCCATCGCGAAAGGCTAA
```

### Output

AMPPred generates the following output files:

1. **CSV Report** (`output_YYYYMMDDHHMMSS.csv`): Contains predictions for all sequences
   - `header`: Sequence identifier
   - `prediction`: Boolean (True for AMP, False for non-AMP)
   - `probability`: Confidence score

2. **AMP FASTA** (`AMP_sequences_YYYYMMDDHHMMSS.fasta`): Contains only predicted AMP sequences (when using `--get_amp`)

### Example Output

```csv
header,prediction,probability
sequence_1,True,0.9234
sequence_2,False,0.1567
sequence_3,True,0.8901
```

## Workflow

1. **Input Processing**: Load nucleotide sequences from FASTA file
2. **Gene Prediction** (optional): Extract coding sequences using Pyrodigal
3. **Sequence Filtering**: Apply length filters (10-200 amino acids by default)
4. **Model Inference**: Use BERT-based model for AMP prediction
5. **Results Export**: Generate CSV report and optional AMP FASTA file

## Model Architecture

AMPPred uses a modified BERT architecture based on the [GENA-LM](https://github.com/AIRI-Institute/GENA_LM) framework, specifically adapted for antimicrobial peptide classification. 

### Model Variants

The default model (`Neeraj0101/AMP-Predict`) is fine-tuned specifically for antimicrobial peptide prediction with:
- Binary classification output (AMP vs non-AMP)
- Pre-trained on large genomic datasets using GENA-LM architecture

### Custom Models

You can use custom models by:

1. Uploading your model to Hugging Face Hub
2. Using the model name with `--model` parameter
3. Ensuring the model follows the same architecture (BertForSequenceClassification)
4. Compatible with the GENA-LM architectural modifications

## Performance Considerations

- **GPU Acceleration**: The tool automatically uses GPU if available
- **Batch Processing**: Sequences are processed in batches for efficiency
- **Memory Management**: Large datasets are processed efficiently with proper memory management

## Troubleshooting

### Common Issues

1. **Model Not Found**: Run `amp-build --model <model-name>` to download the model first
2. **CUDA Errors**: Ensure compatible PyTorch and CUDA versions
3. **Memory Issues**: Reduce batch size or use CPU-only mode for very large datasets
4. **No Sequences Found**: Check FASTA file format and sequence length requirements

## API Usage

You can also use AMPPred programmatically:

```python
from amppred.utility.helper import load_model, get_predictions, read_fasta

# Load model
model, tokenizer = load_model('Neeraj0101/AMP-Predict')

# Read sequences
sequence_df = read_fasta('sequences.fasta')

# Get predictions
predicted_df, sequence_df = get_predictions(model, tokenizer, sequence_df)
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License
Copyright 2025 Neeraj Kumar Singh

This project is licensed under the Apache License 2.0.

## Citation

If you use AMPPred in your research, please cite:

```bibtex
@misc{amppred,
  title={AMPPred: Antimicrobial Peptide Prediction Tool},
  author={Singh, Neeraj Kumar},
  year={2025},
  url={https://github.com/neeraj1014/Antimicrobial-Peptide-Predictor}
}
```

### Acknowledgments

This work builds upon the GENA-LM architecture developed by the AIRI Institute:

```bibtex
@article{fishman2023gena,
  title={GENA-LM: A Family of Open-Source Foundational Models for Long DNA Sequences},
  author={Fishman, Veniamin and Kuratov, Yury and Petrov, Maxim and others},
  journal={arXiv preprint arXiv:2312.10769},
  year={2023},
  url={https://github.com/AIRI-Institute/GENA_LM}
}
```

**Architecture Credits**: The model architecture is based on the GENA-LM framework (https://github.com/AIRI-Institute/GENA_LM), which provides enhanced BERT architectures specifically designed for genomic and biological sequence analysis. The GENA-LM modifications include support for longer sequences, improved positional encodings, and optimizations for biological data processing.

## Support

For questions and support, please:
- Open an issue on GitHub
- Contact: neeraj.nks1001@gmail.com

## Changelog

### v0.1.0
- Initial release
- Basic AMP prediction functionality
- Model management system
- Command-line interface