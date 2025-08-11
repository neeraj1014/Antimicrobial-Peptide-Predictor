import argparse
from amppred.utility.helper import load_model, get_predictions, read_fasta, check_model_names, get_codon_usage, write_prediction_files, filter_sequences
from datetime import datetime
import warnings
import sys
import os
import gc
import torch
from pathlib import Path

warnings.filterwarnings("ignore")

class ProgressTracker:
    """Helper class for tracking progress and timing"""
    def __init__(self):
        self.start_time = datetime.now()
        self.step_times = {}
    
    def log_step(self, step_name, message="", details=None):
        current_time = datetime.now()
        elapsed = current_time - self.start_time
        
        if step_name in self.step_times:
            step_duration = current_time - self.step_times[step_name]
            print(f"✓ {step_name} completed in {step_duration.total_seconds():.2f}s")
        
        if message:
            print(f"→ {message}")
        
        if details:
            print(f"  Details: {details}")
        
        self.step_times[step_name] = current_time
        return elapsed.total_seconds()

def validate_inputs(args):
    """Validate input arguments and files"""
    print("Validating inputs...")
    
    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    # Check file size
    file_size = Path(args.input).stat().st_size
    print(f"  Input file size: {file_size / (1024*1024):.2f} MB")
    
    # Validate filter parameters
    if args.filter_min and args.filter_max and args.filter_min > args.filter_max:
        print(f"Error: min_length ({args.filter_min}) cannot be greater than max_length ({args.filter_max})")
        sys.exit(1)
    
    # Check available memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU memory available: {gpu_memory:.2f} GB")
    
    print("✓ Input validation completed")

def optimize_environment():
    """Optimize environment for processing"""
    print("Optimizing environment...")
    
    # Set environment variables for performance
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = str(min(8, os.cpu_count()))
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  GPU cache cleared")
    
    # Force garbage collection
    gc.collect()
    print("✓ Environment optimization completed")

def main():
    parser = argparse.ArgumentParser(
        description="AMP Prediction Tool - Efficient antimicrobial peptide prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py --input sequences.fasta --model Neeraj0101/AMP-Predict --get_amp
  python script.py --input genome.fasta --codon --filter_min 30 --filter_max 150
        """
    )
    
    parser.add_argument("--input", type=str, required=True, 
                       help="Input FASTA file containing nucleotide sequences")
    parser.add_argument("--codon", action="store_true", 
                       help="Extract coding sequences using gene prediction")
    parser.add_argument("--model", type=str, default='Neeraj0101/AMP-Predict', 
                       help="Model name or path (default: Neeraj0101/AMP-Predict)")
    parser.add_argument("--get_amp", action="store_true", 
                       help="Generate FASTA file with predicted AMP sequences")
    parser.add_argument("--filter_min", type=int, default=10, 
                       help="Minimum sequence length for filtering (default: 10)")
    parser.add_argument("--filter_max", type=int, default=200, 
                       help="Maximum sequence length for filtering (default: 200)")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for model inference (default: 32)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    
    # Initialize progress tracker
    tracker = ProgressTracker()
    
    print("=" * 60)
    print("AMP Prediction Tool Started")
    print("=" * 60)
    
    # Validate inputs
    validate_inputs(args)
    
    # Optimize environment
    optimize_environment()
    
    # Check model availability
    print("Checking model availability...")
    if not check_model_names(args.model):
        print(f"Model '{args.model}' not found in model registry.")
        print("Please run 'AMPPred-build' to download the model first.")
        sys.exit(1)
    print(f"Model '{args.model}' is available")
    
    sequence_df = None
    total_sequences = 0
    coding_sequences = 0
    
    try:
        # Step 1: Read FASTA file
        tracker.log_step("file_reading", "Reading FASTA file...")
        sequence_df = read_fasta(args.input)
        
        if sequence_df.empty:
            print("No sequences found in the input file")
            sys.exit(1)
        
        total_sequences = len(sequence_df)
        avg_length = sequence_df['length'].mean()
        tracker.log_step("file_reading", 
                         f"Successfully loaded {total_sequences:,} sequences",
                         f"Average length: {avg_length:.1f} bp")
        
        # Step 2: Process sequences (codon extraction or direct use)
        if not args.codon:
            tracker.log_step("codon_extraction", "Extracting coding sequences...")
            original_count = len(sequence_df)
            sequence_df = get_codon_usage(sequence_df)
            
            if sequence_df.empty:
                print("No coding sequences found after gene prediction")
                sys.exit(1)
            
            coding_sequences = len(sequence_df)
            tracker.log_step("codon_extraction", 
                             f"Extracted {coding_sequences:,} coding sequences from {original_count:,} input sequences")
        else:
            tracker.log_step("direct_processing", "Using sequences directly for prediction")
        
        # Step 3: Filter sequences
        tracker.log_step("filtering", "Filtering sequences by length...")
        pre_filter_count = len(sequence_df)
        sequence_df = filter_sequences(sequence_df, args.filter_min, args.filter_max)
        
        if sequence_df.empty:
            print(f"No sequences remain after filtering (length: {args.filter_min}-{args.filter_max})")
            sys.exit(1)
        
        post_filter_count = len(sequence_df)
        filtered_out = pre_filter_count - post_filter_count
        tracker.log_step("filtering", 
                         f"Retained {post_filter_count:,} sequences after filtering",
                         f"Filtered out: {filtered_out:,} sequences ({(filtered_out/pre_filter_count)*100:.1f}%)")
        
        # Step 4: Load model
        tracker.log_step("model_loading", "Loading prediction model...")
        model, tokenizer = load_model(args.model)
        
        # Model info
        model_params = sum(p.numel() for p in model.parameters())
        device = next(model.parameters()).device
        tracker.log_step("model_loading", 
                         f"Model loaded successfully on {device}",
                         f"Parameters: {model_params:,}")
        
        # Step 5: Run predictions
        tracker.log_step("prediction", "Running AMP predictions...")
        predicted_df, sequence_df = get_predictions(model, tokenizer, sequence_df, 
                                                   batch_size=args.batch_size)
        
        # Calculate prediction statistics
        total_predictions = len(predicted_df)
        amp_predictions = predicted_df['prediction'].sum()
        amp_percentage = (amp_predictions / total_predictions) * 100
        avg_confidence = predicted_df['probability'].mean()
        
        tracker.log_step("prediction", 
                         f"Predictions completed for {total_predictions:,} sequences",
                         f"AMP predictions: {amp_predictions:,} ({amp_percentage:.1f}%), Avg confidence: {avg_confidence:.3f}")
        
        # Step 6: Write output files
        tracker.log_step("output_writing", "Writing output files...")
        write_prediction_files(sequence_df, predicted_df, args.get_amp)
        tracker.log_step("output_writing", "Output files written successfully")
        
        # Final summary
        total_time = tracker.log_step("complete", "")
        print("\n" + "=" * 60)
        print("AMP Prediction Completed Successfully!")
        print("=" * 60)
        print(f"Processing Summary:")
        print(f"   • Total processing time: {total_time:.2f} seconds")
        print(f"   • Input sequences: {total_sequences:,}")
        print(f"   • Coding sequences: {coding_sequences:,}")
        print(f"   • Sequences processed: {post_filter_count:,}")
        print(f"   • AMP predictions: {amp_predictions:,} ({amp_percentage:.1f}%)")
        print(f"   • Average confidence: {avg_confidence:.3f}")
        
        if args.get_amp and amp_predictions > 0:
            print(f"   • AMP FASTA file generated with {amp_predictions:,} sequences")
        
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()