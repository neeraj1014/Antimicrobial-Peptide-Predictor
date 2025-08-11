import argparse
from amppred.utility.helper import load_model,get_predictions,read_fasta,check_model_names,get_codon_usage,write_prediction_files
import builtins
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Run MyTool on sequences")
    parser.add_argument("--input", type=str, required=True, help="Input file nucleotides FASTA ")
    parser.add_argument("--codon", action="store_true", help="Model name or path")
    parser.add_argument("--model", type=str, default='Neeraj0101/AMP-Predict', help="Model name or path")
    parser.add_argument("--get_amp", action="store_true", help="Get AMP sequences")
    
    args = parser.parse_args()
    original_print = builtins.print
    def print_with_time(*args, **kwargs):
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        original_print(f"[{time_str}]", *args, **kwargs)

    builtins.print = print_with_time
    
    if not check_model_names(args.model):
        print(f"Model {args.model} not found in model names. Please run amp-build to download model.")
        return

    if args.codon:
        print("Reading FASTA file for predictions...")
        sequence_df = read_fasta(args.input)
    else:
        print("Reading FASTA file for codon extraction...")
        sequence_df = read_fasta(args.input)
        sequence_df = get_codon_usage(sequence_df)
        print("Codon extracted from sequences...")
        
        
    
    if not sequence_df.empty:
        print("Loading model for predictions...")        
        model, tokenizer = load_model(args.model)
        print("Model loaded successfully...")
        print("Getting predictions for sequences...")
        predicted_df, sequence_df = get_predictions(model, tokenizer, sequence_df)
        print("Predictions obtained successfully...")
        write_prediction_files(sequence_df, predicted_df,args.get_amp)
        print("Prediction files written successfully...")
    else:
        if args.codon:
            print("No sequences found in the input file for predictions.")
        else:
            print("No codon sequences found for predictions.")


if __name__ == "__main__":
    main()