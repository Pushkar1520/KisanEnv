import wandb
import sys

def download_metrics(run_id: str):
    print(f"Connecting to Weights & Biases API...")
    api = wandb.Api()

    # The project path based on your snippet
    run_path = f"pushkarbatrab-nims-university/huggingface/{run_id}"
    
    try:
        print(f"Fetching data for run: {run_path}")
        run = api.run(run_path)
        
        print("Downloading training history...")
        metrics_dataframe = run.history()
        
        output_file = "qwen_training_metrics.csv"
        metrics_dataframe.to_csv(output_file, index=False)
        print(f"✅ Success! Training data saved to: {output_file}")
        
    except wandb.errors.CommError:
        print(f"❌ Error: Could not find run '{run_id}'.")
        print("Please check your run_id and ensure you are logged in using 'wandb login'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_wandb_data.py <your_run_id>")
        print("Example: python download_wandb_data.py 1a2b3c4d")
        sys.exit(1)
        
    run_id = sys.argv[1]
    download_metrics(run_id)
