import argparse
from amppred.utility.helper import check_if_model_exists,check_model_names,load_model,get_model_names,add_model_name
import builtins
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Build or download MyTool model files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        type=str,
        default='Neeraj0101/AMP-Predict',
        help="Model name to be imported or downloaded. If the model is not available, it will be downloaded from Hugging Face.",
    )
    group.add_argument("--list_models", action="store_true", help="Get list of models")
    args = parser.parse_args()
    original_print = builtins.print
    def print_with_time(*args, **kwargs):
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        original_print(f"[{time_str}]", *args, **kwargs)

    builtins.print = print_with_time
    # Check if the model exists in the Hugging Face Hub
    if args.list_models:
        model_list = get_model_names()
        if model_list:
            print(f"Available models: {"|".join(model_list)}")
        else:
            print("No models found. Please run amppred-build or amppred-build --model custom_models to download models.")
    else:
        if check_if_model_exists(args.model):
            if check_model_names(args.model):
                print(f"Model {args.model} already exists locally.")
            else:
                load_model(args.model)
                add_model_name(args.model)
                print(f"Model {args.model} is downloaded.")
        else:
            print(f"Model {args.model} does not exist in Hugging Face Hub. Please check the model name.")
            # Optionally, you can implement logic to download the model if it doesn't exist
            # For now, we just print a message
            # load_model(args.model)  # Uncomment if you want to attempt loading the model anyway   
    


        
if __name__ == "__main__":
    main()
