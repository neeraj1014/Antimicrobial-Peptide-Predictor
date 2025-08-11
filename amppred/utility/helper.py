from amppred.utility.modeling_bert import BertForSequenceClassification
from datetime import datetime
from transformers import AutoTokenizer
import pandas as pd
from Bio import SeqIO
import pyrodigal
import json
import os


# utility.py
def load_model(model_name):
    """
    Load the pre-trained model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

def classify_prediction(prediction):
        if prediction == 0:
            return False
        elif prediction == 1:
            return True
        
def get_predictions(model, tokenizer, sequence_df):
    """
    Get predictions for the input text using the pre-trained model.
    """
    sequences = sequence_df['sequence'].tolist()
    headers = sequence_df['header'].tolist()
    # Tokenize all sequences together
    inputs = tokenizer(
        sequences,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=256
    )
    outputs = model(**inputs)
    logits = outputs.logits.softmax(dim=-1)
    predictions = logits.argmax(dim=-1).tolist()

    predicted_df = pd.DataFrame({
        'header': headers,
        'prediction': predictions,
        'probability': logits.max(dim=-1).values.tolist()})
    predicted_df['prediction'] = predicted_df['prediction'].apply(classify_prediction)

    return predicted_df,sequence_df

def read_fasta(fasta_file):
    """
    Read a FASTA file and return a list of sequences.
    """
    megares_seqs = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        accession = record.id
        megares_seqs.append({
            "header": accession,
            "sequence": str(record.seq.upper()),
            "length": len(record.seq),
            "description": record.description if record.description else accession
        })
    sequence_df = pd.DataFrame(megares_seqs)
    return sequence_df

def write_prediction_files(sequence_df, predicted_df, get_amp):
    """
    Write sequences to a FASTA file.
    """
    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f'AMP_sequences_{time_str}.fasta'
    output_file_name = f'output_{time_str}.csv'
    
    sequence_df = sequence_df.merge(predicted_df, on='header', how='left')
    sequence_df = sequence_df.loc[sequence_df['prediction'] > 0].reset_index(drop=True)
    predicted_df.to_csv(output_file_name, index=False)
    print(f"Total AMP sequences found: {len(sequence_df)}")
    if get_amp:
        if sequence_df.empty:
            print("No AMP sequences found. So, no FASTA file generated.")
        else:
            with open(output_file, 'w') as f:
                for _, row in sequence_df.iterrows():
                    f.write(f">{row['description']}\n{row['sequence']}\n")
    
def get_codon_usage(sequence_df):
    """
    Calculate codon usage for each sequence in the DataFrame.
    """
    gene_finder = pyrodigal.GeneFinder(meta=True)
    coding_regions = []
    # Run gene prediction on one sequence
    for _ , row in sequence_df.iterrows():
        seq = row['sequence']
        genes = gene_finder.find_genes(seq)

        for i, gene in enumerate(genes):
            cds_seq = gene.sequence()
            coding_regions.append({
                "header": row['header'],
                "gene_id": f"{row['header']}_gene_{i+1}",
                "sequence": cds_seq,
                "length": len(cds_seq),
                "description": f"{row['description']}| gene_{i+1}"
            })
    coding_df = pd.DataFrame(coding_regions)
    return coding_df

def filter_sequences(sequence_df, min_length=10, max_length=200):
    """
    Filter sequences based on a minimum length.
    """
    if max_length and min_length:
        filtered_df = sequence_df[(sequence_df['length'] >= min_length)&(sequence_df['length'] <= max_length)].reset_index(drop=True)
    elif min_length:
        filtered_df = sequence_df[sequence_df['length'] >= min_length].reset_index(drop=True)
    elif max_length:
        filtered_df = sequence_df[sequence_df['length'] <= max_length].reset_index(drop=True)
    else:
        filtered_df = sequence_df.reset_index(drop=True)
        
    return filtered_df

def check_if_model_exists(model_name):
    """
    Check if the model exists in the Hugging Face Hub.
    """
    from huggingface_hub import model_info
    try:
        model_info(model_name)
        return True
    except Exception as e:
        print(f"Model {model_name} does not exist: {e}")
        return False
def add_model_name(model_name):
    """
    Add the model name to the model_names.json file.
    """
    current_script_path = os.path.realpath(__file__)
    file_path= os.path.join(os.path.dirname(current_script_path), 'model_names.json')

    if not os.path.exists(file_path):
        model_check = {
            "models": []
        }
        with open(file_path, 'w') as f:
            json.dump(model_check, f, indent=4)

    with open(file_path, 'r') as f:
        model_names = json.load(f)

    if model_name not in model_names['models']:
        model_names['models'].append(model_name)
        with open(file_path, 'w') as f:
            json.dump(model_names, f, indent=4)
            
def check_model_names(model_for):
    """
    Get the list of available model names from the Hugging Face Hub.
    """
    current_script_path = os.path.realpath(__file__)
    file_path= os.path.join(os.path.dirname(current_script_path), 'model_names.json')

    if not os.path.exists(file_path):
        model_check = {
            "models": []
        }
        with open(file_path, 'w') as f:
            json.dump(model_check, f, indent=4)
    else:
        with open(file_path, 'r') as f:
            model_names = json.load(f)
        if model_for in model_names['models']:
            return True
        else:
            return False
def get_model_names():
    """
    Get the list of available model names from the Hugging Face Hub.
    """
    current_script_path = os.path.realpath(__file__)
    file_path= os.path.join(os.path.dirname(current_script_path), 'model_names.json')

    if not os.path.exists(file_path):
        return []
    else:
        with open(file_path, 'r') as f:
            model_names = json.load(f)
        return model_names['models']



    