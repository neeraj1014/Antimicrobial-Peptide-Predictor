import argparse
from amppred.utility.helper import check_if_model_exists, check_model_names, load_model, get_model_names, add_model_name
import builtins
from datetime import datetime
import warnings
import sys
import os
import gc
import torch
from pathlib import Path
from huggingface_hub import model_info, HfApi
import requests

warnings.filterwarnings("ignore")

class ModelBuildTracker:
    """Helper class for tracking model build progress and timing"""
    def __init__(self):
        self.start_time = datetime.now()
        self.step_times = {}
    
    def log_step(self, step_name, message="", details=None, status="info"):
        current_time = datetime.now()
        elapsed = current_time - self.start_time
        
        status_icons = {
            "info": "→",
            "success": "✓", 
            "error": "❌",
            "warning": "⚠️",
            "downloading": "⬇️",
            "checking": "🔍"
        }
        
        icon = status_icons.get(status, "→")
        
        if step_name in self.step_times:
            step_duration = current_time - self.step_times[step_name]
            if status == "success":
                print(f"{icon} {step_name} completed in {step_duration.total_seconds():.2f}s")
        
        if message:
            print(f"{icon} {message}")
        
        if details:
            if isinstance(details, dict):
                for key, value in details.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  Details: {details}")
        
        self.step_times[step_name] = current_time
        return elapsed.total_seconds()

def get_model_info_safe(model_name):
    """Safely get model information with detailed error handling"""
    try:
        info = model_info(model_name)
        return {
            "exists": True,
            "size": getattr(info, 'size', 'Unknown'),
            "downloads": getattr(info, 'downloads', 0),
            "likes": getattr(info, 'likes', 0),
            "tags": getattr(info, 'tags', []),
            "last_modified": getattr(info, 'lastModified', 'Unknown')
        }
    except Exception as e:
        return {
            "exists": False,
            "error": str(e)
        }

def format_size(size_bytes):
    """Format file size in human readable format"""
    if not isinstance(size_bytes, (int, float)) or size_bytes == 0:
        return "Unknown"
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def validate_model_name(model_name):
    """Validate model name format"""
    if not model_name:
        return False, "Model name cannot be empty"
    
    # Check basic format (should contain '/' for Hugging Face models)
    if '/' not in model_name:
        return False, "Model name should be in format 'username/model-name'"
    
    parts = model_name.split('/')
    if len(parts) != 2:
        return False, "Model name should contain exactly one '/'"
    
    username, modelname = parts
    if not username or not modelname:
        return False, "Both username and model name parts are required"
    
    return True, "Valid model name format"

def check_internet_connection():
    """Check if internet connection is available"""
    try:
        response = requests.get("https://huggingface.co", timeout=5)
        return response.status_code == 200
    except:
        return False

def optimize_environment_for_download():
    """Optimize environment for model downloading"""
    print("Optimizing environment for download...")
    
    # Set environment variables for better download performance
    os.environ['HF_HUB_CACHE'] = os.path.expanduser('~/.cache/huggingface/hub')
    os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser('~/.cache/huggingface/transformers')
    
    # Create cache directories if they don't exist
    cache_dirs = [
        os.environ.get('HF_HUB_CACHE'),
        os.environ.get('TRANSFORMERS_CACHE')
    ]
    
    for cache_dir in cache_dirs:
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  GPU cache cleared")
    
    gc.collect()
    print("✓ Environment optimization completed")

def display_model_list(model_list, tracker):
    """Display available models in a formatted table"""
    if not model_list:
        print("No models found in local registry")
        print("   Run 'amppred-build --model <model_name>' to download models")
        return
    
    print(f"Available Models ({len(model_list)} total):")
    print("-" * 80)
    print(f"{'Index':<6} {'Model Name':<40} {'Status':<15} {'Info'}")
    print("-" * 80)
    
    for i, model_name in enumerate(model_list, 1):
        # Check if model files actually exist locally
        try:
            # This is a simple check - you might want to implement more robust checking
            status = "✓ Available" if check_model_names(model_name) else "⚠️ Registry Only"
            
            # Get additional info if possible
            model_info_data = get_model_info_safe(model_name)
            info = ""
            if model_info_data["exists"]:
                downloads = model_info_data.get("downloads", 0)
                if downloads > 0:
                    info = f"{downloads:,} downloads"
            
            print(f"{i:<6} {model_name:<40} {status:<15} {info}")
            
        except Exception as e:
            print(f"{i:<6} {model_name:<40} {'❌ Error':<15} {str(e)[:20]}...")
    
    print("-" * 80)

def main():
    parser = argparse.ArgumentParser(
        description="AMP Model Builder - Download and manage AMP prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  AMPPred-build --model Neeraj0101/AMP-Predict          # Download specific model
  AMPPred-build --model username/custom-amp-model       # Download custom model
  AMPPred-build --list_models                           # List available models
  AMPPred-build --model local_model --force             # Force re-download
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        type=str,
        help="Model name to download from Hugging Face Hub (format: username/model-name)"
    )
    group.add_argument(
        "--list_models", 
        action="store_true", 
        help="Display all available models in local registry"
    )
    
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force re-download even if model exists locally"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output with detailed information"
    )
    parser.add_argument(
        "--no_cache", 
        action="store_true", 
        help="Disable caching during model download"
    )
    
    args = parser.parse_args()
    
    # Initialize progress tracker
    tracker = ModelBuildTracker()
    
    print("=" * 70)
    print("AMP Model Builder Started")
    print("=" * 70)
    
    try:
        # Handle list models command
        if args.list_models:
            tracker.log_step("list_models", "Retrieving model list...", status="checking")
            model_list = get_model_names()
            tracker.log_step("list_models", status="success")
            display_model_list(model_list, tracker)
            
            if not model_list:
                print("\nTip: Download your first model using:")
                print("   amppred-build --model Neeraj0101/AMP-Predict")
            
            total_time = tracker.log_step("complete", "")
            print(f"\nTotal time: {total_time:.2f} seconds")
            return
        
        # Handle model download
        model_name = args.model
        if not model_name:
            model_name = 'Neeraj0101/AMP-Predict'
            print(f"No model specified, using default: {model_name}")
        
        # Validate model name format
        tracker.log_step("validation", "Validating model name...", status="checking")
        is_valid, validation_msg = validate_model_name(model_name)
        if not is_valid:
            tracker.log_step("validation", f"Invalid model name: {validation_msg}", status="error")
            sys.exit(1)
        tracker.log_step("validation", "Model name format is valid", status="success")
        
        # Check internet connection
        tracker.log_step("connectivity", "Checking internet connection...", status="checking")
        if not check_internet_connection():
            tracker.log_step("connectivity", "No internet connection available", status="error")
            print("   Please check your network connection and try again")
            sys.exit(1)
        tracker.log_step("connectivity", "Internet connection confirmed", status="success")
        
        # Check if model exists locally
        tracker.log_step("local_check", "Checking local model registry...", status="checking")
        model_exists_locally = check_model_names(model_name)
        
        if model_exists_locally and not args.force:
            tracker.log_step("local_check", f"Model '{model_name}' already exists locally", status="success")
            print("   Use --force flag to re-download the model")
            return
        elif model_exists_locally and args.force:
            tracker.log_step("local_check", f"Model exists locally but force re-download requested", status="warning")
        else:
            tracker.log_step("local_check", f"Model not found locally, will download", status="info")
        
        # Check if model exists on Hugging Face Hub
        tracker.log_step("hub_check", "Checking model availability on Hugging Face Hub...", status="checking")
        
        if args.verbose:
            print("   Fetching model metadata...")
        
        model_info_data = get_model_info_safe(model_name)
        
        if not model_info_data["exists"]:
            tracker.log_step("hub_check", f"Model '{model_name}' not found on Hugging Face Hub", status="error")
            print(f"   Error: {model_info_data.get('error', 'Unknown error')}")
            print("   Please verify the model name and try again")
            sys.exit(1)
        
        # Display model information
        model_details = {
            "Model Size": format_size(model_info_data.get("size")),
            "Downloads": f"{model_info_data.get('downloads', 0):,}",
            "Likes": f"{model_info_data.get('likes', 0):,}",
            "Tags": ", ".join(model_info_data.get("tags", [])[:3])  # Show first 3 tags
        }
        
        tracker.log_step("hub_check", f"Model '{model_name}' found on Hugging Face Hub", 
                         details=model_details, status="success")
        
        # Optimize environment for download
        optimize_environment_for_download()
        
        # Download and load the model
        tracker.log_step("download", f"Downloading model '{model_name}'...", status="downloading")
        print("   This may take several minutes depending on model size and connection speed...")
        
        try:
            # Set cache options
            if args.no_cache:
                os.environ['TRANSFORMERS_OFFLINE'] = '0'
                os.environ['HF_HUB_OFFLINE'] = '0'
            
            # Load the model (this will download it if not present)
            model, tokenizer = load_model(model_name)
            
            # Get model statistics
            model_params = sum(p.numel() for p in model.parameters()) if model else 0
            device = str(next(model.parameters()).device) if model else "unknown"
            
            download_details = {
                "Parameters": f"{model_params:,}",
                "Device": device,
                "Tokenizer vocab size": f"{len(tokenizer.vocab):,}" if hasattr(tokenizer, 'vocab') else "Unknown"
            }
            
            tracker.log_step("download", f"Model downloaded and loaded successfully", 
                             details=download_details, status="success")
            
        except Exception as e:
            tracker.log_step("download", f"Failed to download model: {str(e)}", status="error")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        
        # Add model to registry
        tracker.log_step("registry", "Adding model to local registry...", status="info")
        add_model_name(model_name)
        tracker.log_step("registry", "Model added to registry successfully", status="success")
        
        # Final summary
        total_time = tracker.log_step("complete", "")
        print("\n" + "=" * 70)
        print("Model Build Completed Successfully!")
        print("=" * 70)
        print(f"Build Summary:")
        print(f"   • Model: {model_name}")
        print(f"   • Parameters: {model_params:,}")
        print(f"   • Size: {model_details['Model Size']}")
        print(f"   • Total time: {total_time:.2f} seconds")
        print(f"   • Status: Ready for prediction")
        print("\nUsage:")
        print(f"   AMPPred --input sequences.fasta --model {model_name}")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
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