from amppred.utility.modeling_bert import BertForSequenceClassification
from datetime import datetime
from transformers import AutoTokenizer
import pandas as pd
from Bio import SeqIO
import pyrodigal
import json
import os
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path

# utility.py
def load_model(model_name):
    """
    Load the pre-trained model and tokenizer with optimization.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    
    # Optimize model for inference
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        # Enable mixed precision if available
        model = model.half() if hasattr(model, 'half') else model
    
    return model, tokenizer

def classify_prediction(prediction):
    """Vectorized classification function"""
    return prediction == 1

def get_predictions(model, tokenizer, sequence_df, batch_size=32, device=None):
    """
    Advanced batch processing with device management and memory optimization.
    
    Args:
        model: Pre-trained model
        tokenizer: Tokenizer for the model
        sequence_df: DataFrame containing 'sequence' and 'header' columns
        batch_size: Number of sequences to process in each batch (default: 32)
        device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        tuple: (predicted_df, sequence_df)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    sequences = sequence_df['sequence'].tolist()
    headers = sequence_df['header'].tolist()
    
    all_predictions = []
    all_probabilities = []
    
    # Process data in batches with optimizations
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches", leave=False):
            # Get current batch
            batch_sequences = sequences[i:i + batch_size]
            
            # Tokenize current batch with optimized settings
            inputs = tokenizer(
                batch_sequences,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=256,
                return_attention_mask=True,
                add_special_tokens=True
            )
            
            # Move inputs to device efficiently
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
            
            # Get predictions for current batch
            outputs = model(**inputs)
            logits = outputs.logits.softmax(dim=-1)
            batch_predictions = logits.argmax(dim=-1).cpu().numpy()  # Use numpy for efficiency
            batch_probabilities = logits.max(dim=-1).values.cpu().numpy()
            
            # Store batch results
            all_predictions.extend(batch_predictions.tolist())
            all_probabilities.extend(batch_probabilities.tolist())
            
            # Clear GPU cache after each batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Create results DataFrame with vectorized operations
    predicted_df = pd.DataFrame({
        'header': headers,
        'prediction': all_predictions,
        'probability': all_probabilities
    })
    
    # Use vectorized apply for classification
    predicted_df['prediction'] = predicted_df['prediction'].apply(classify_prediction)
    
    return predicted_df, sequence_df

def read_fasta(fasta_file):
    """
    Read a FASTA file and return a list of sequences with optimized parsing.
    """
    megares_seqs = []
    
    # Use generator for memory efficiency with large files
    try:
        records = SeqIO.parse(fasta_file, "fasta")
        for record in records:
            accession = record.id
            sequence = str(record.seq.upper())
            megares_seqs.append({
                "header": accession,
                "sequence": sequence,
                "length": len(sequence),
                "description": record.description if record.description else accession
            })
    except Exception as e:
        print(f"Error reading FASTA file {fasta_file}: {e}")
        return pd.DataFrame()
    
    # Create DataFrame with optimized dtypes
    sequence_df = pd.DataFrame(megares_seqs)
    if not sequence_df.empty:
        sequence_df = sequence_df.astype({
            'header': 'string',
            'sequence': 'string', 
            'length': 'int32',
            'description': 'string'
        })
    
    return sequence_df

def write_prediction_files(sequence_df, predicted_df, get_amp):
    """
    Write sequences to a FASTA file with optimized I/O operations.
    """
    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f'AMP_sequences_{time_str}.fasta'
    output_file_name = f'output_{time_str}.csv'
    
    # Efficient merge operation
    sequence_df = sequence_df.merge(predicted_df, on='header', how='left')
    
    # Vectorized filtering
    amp_sequences = sequence_df.loc[sequence_df['prediction'] == True].reset_index(drop=True)
    
    # Write CSV with optimized parameters
    predicted_df.to_csv(output_file_name, index=False, encoding='utf-8')
    
    print(f"Total AMP sequences found: {len(amp_sequences)}")
    
    if get_amp:
        if amp_sequences.empty:
            print("No AMP sequences found. So, no FASTA file generated.")
        else:
            # Optimized file writing using list comprehension and join
            fasta_lines = []
            for _, row in amp_sequences.iterrows():
                fasta_lines.extend([f">{row['description']}", row['sequence']])
            
            with open(output_file, 'w', encoding='utf-8', buffering=8192) as f:
                f.write('\n'.join(fasta_lines) + '\n')

def get_codon_usage(sequence_df):
    """
    Calculate codon usage for each sequence in the DataFrame with optimizations.
    """
    # Initialize gene finder once
    gene_finder = pyrodigal.GeneFinder(meta=True)
    coding_regions = []
    
    # Process sequences with progress bar
    for _, row in tqdm(sequence_df.iterrows(), total=len(sequence_df), desc="Processing sequences", leave=False):
        seq = row['sequence']
        
        try:
            genes = gene_finder.find_genes(seq)
            
            # Use list comprehension for efficiency
            for i, gene in enumerate(genes):
                cds_seq = gene.sequence()
                coding_regions.append({
                    "header": row['header'],
                    "gene_id": f"{row['header']}_gene_{i+1}",
                    "sequence": cds_seq,
                    "length": len(cds_seq),
                    "description": f"{row['description']}| gene_{i+1}"
                })
        except Exception as e:
            print(f"Error processing sequence {row['header']}: {e}")
            continue
    
    # Create DataFrame with optimized dtypes
    coding_df = pd.DataFrame(coding_regions)
    if not coding_df.empty:
        coding_df = coding_df.astype({
            'header': 'string',
            'gene_id': 'string',
            'sequence': 'string',
            'length': 'int32',
            'description': 'string'
        })
    
    return coding_df

def filter_sequences(sequence_df, min_length=10, max_length=200):
    """
    Filter sequences based on length with vectorized operations.
    """
    if sequence_df.empty:
        return sequence_df
    
    # Create boolean mask for efficient filtering
    mask = pd.Series([True] * len(sequence_df), index=sequence_df.index)
    
    if min_length is not None:
        mask &= (sequence_df['length'] >= min_length)
    
    if max_length is not None:
        mask &= (sequence_df['length'] <= max_length)
    
    filtered_df = sequence_df[mask].reset_index(drop=True)
    return filtered_df

def check_if_model_exists(model_name):
    """
    Check if the model exists in the Hugging Face Hub with caching.
    """
    try:
        from huggingface_hub import model_info
        model_info(model_name)
        return True
    except ImportError:
        print("huggingface_hub not installed. Cannot verify model existence.")
        return False
    except Exception as e:
        print(f"Model {model_name} does not exist: {e}")
        return False

def add_model_name(model_name):
    """
    Add the model name to the model_names.json file with optimized I/O.
    """
    current_script_path = Path(__file__).resolve()
    file_path = current_script_path.parent / 'model_names.json'
    
    # Initialize default structure
    default_structure = {"models": []}
    
    # Read existing data or create new
    if file_path.exists():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                model_names = json.load(f)
        except (json.JSONDecodeError, IOError):
            model_names = default_structure
    else:
        model_names = default_structure
    
    # Add model if not exists
    if model_name not in model_names.get('models', []):
        model_names['models'].append(model_name)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(model_names, f, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"Error writing to model_names.json: {e}")

def check_model_names(model_for):
    """
    Check if model exists in the list with optimized file handling.
    """
    current_script_path = Path(__file__).resolve()
    file_path = current_script_path.parent / 'model_names.json'
    
    if not file_path.exists():
        # Create file with default structure
        default_structure = {"models": []}
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(default_structure, f, indent=4)
        except IOError:
            pass
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            model_names = json.load(f)
        return model_for in model_names.get('models', [])
    except (json.JSONDecodeError, IOError):
        return False

def get_model_names():
    """
    Get the list of available model names with optimized file handling.
    """
    current_script_path = Path(__file__).resolve()
    file_path = current_script_path.parent / 'model_names.json'
    
    if not file_path.exists():
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            model_names = json.load(f)
        return model_names.get('models', [])
    except (json.JSONDecodeError, IOError):
        return []