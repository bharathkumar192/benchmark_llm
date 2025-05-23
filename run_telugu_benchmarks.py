#!/usr/bin/env python
# run_telugu_benchmarks.py

import os
import argparse
import json
import datetime
import subprocess
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Telugu language benchmarks on a fine-tuned model")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to your fine-tuned model or model identifier on Hugging Face")
    parser.add_argument("--model_type", type=str, default="hf",
                        choices=["hf", "vllm"],
                        help="Model type to use for evaluation (hf or vllm)")
    parser.add_argument("--tasks", type=str, default="telugu_sentiment,mmlu_telugu",
                        help="Comma-separated list of tasks to evaluate")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument("--num_fewshot", type=int, default=0,
                        help="Number of examples to use for few-shot evaluation")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run evaluation on")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="Trust remote code from Hugging Face")
    
    return parser.parse_args()

def setup_task_configs():
    """
    Ensure custom task configs are in the right location
    """
    src_dir = Path(".")
    tasks_dir = Path("lm-evaluation-harness/lm_eval/tasks/custom_tasks")
    
    # Get absolute path to the tasks_data directory
    tasks_data_path = os.path.abspath("tasks_data")
    
    os.makedirs(tasks_dir, exist_ok=True)
    
    # Create the task YAML files directly
    telugu_sentiment_yaml = f"""
task: telugu_sentiment
dataset_path: {tasks_data_path}/telugu_sentiment
output_type: multiple_choice
test_split: test
validation_split: validation
doc_to_text: "{{{{text}}}}\\nSentiment: "
doc_to_choice: ["positive", "negative", "neutral"]
doc_to_target: label
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
"""

    mmlu_telugu_yaml = f"""
task: mmlu_telugu
dataset_path: {tasks_data_path}/mmlu_telugu
output_type: multiple_choice
test_split: test
validation_split: validation
num_fewshot: 0
doc_to_text: "{{{{question}}}}\\nA. {{{{choices[0]}}}}\\nB. {{{{choices[1]}}}}\\nC. {{{{choices[2]}}}}\\nD. {{{{choices[3]}}}}\\nAnswer:"
doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: answer
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
"""

    # Write the files
    with open(tasks_dir / "telugu_sentiment.yaml", "w") as f:
        f.write(telugu_sentiment_yaml)
    
    with open(tasks_dir / "mmlu_telugu.yaml", "w") as f:
        f.write(mmlu_telugu_yaml)
    
    print(f"Created task config files in {tasks_dir}")

def run_benchmark(args):
    """
    Run the benchmark evaluations
    """
    # Setup task configs
    setup_task_configs()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(args.model_path)
    results_dir = os.path.join(args.output_dir, f"{model_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Build the command with memory offloading parameters
    model_args = f"pretrained={args.model_path}"
    if args.trust_remote_code:
        model_args += ",trust_remote_code=True"
    
    # Add device mapping for better memory utilization
    model_args += ",max_length=2048"
    
    # Set environment variable for PyTorch memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Change directory to the lm-evaluation-harness folder
    original_dir = os.getcwd()
    os.chdir("lm-evaluation-harness")
    
    # To debug the format of the dataset
    print("Debugging dataset format:")
    os.system("python -c \"from datasets import load_dataset; print(load_dataset('csv', data_files='../tasks_data/telugu_sentiment/test.csv')[0][0])\"")
    
    # Run each task separately to isolate issues
    for task in ["telugu_sentiment", "mmlu_telugu"]:
        try:
            # Construct command as a single string without line breaks within quotes
            if task == "mmlu_telugu":
                # Use lower batch size for MMLU to avoid OOM
                cmd = (f"lm_eval --model hf --model_args {model_args} --tasks {task} "
                      f"--device {args.device} --batch_size 1 --num_fewshot 0 "
                      f"--output_path ../{os.path.join(results_dir, f'{task}_results.json')} "
                      f"--include_path lm_eval/tasks/custom_tasks --log_samples")
            else:
                # For sentiment task
                cmd = (f"lm_eval --model hf --model_args {model_args} --tasks {task} "
                      f"--device {args.device} --batch_size {args.batch_size} --num_fewshot {args.num_fewshot} "
                      f"--output_path ../{os.path.join(results_dir, f'{task}_results.json')} "
                      f"--include_path lm_eval/tasks/custom_tasks --log_samples")
            
            print(f"Running benchmark for {task} with command:\n{cmd}")
            subprocess.run(cmd, shell=True, check=True)
            print(f"Successfully completed evaluation for {task}")
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {task}: {e}")
    
    # Change back to the original directory
    os.chdir(original_dir)
    
    # Generate a summary report for successful tasks
    try:
        generate_summary(results_dir, args)
    except Exception as e:
        print(f"Error generating summary: {e}")
    
    return results_dir

def generate_summary(results_dir, args):
    """
    Generate a human-readable summary of the benchmark results
    """
    results_file = os.path.join(results_dir, 'results.json')
    if not os.path.exists(results_file):
        print(f"Warning: Results file not found at {results_file}")
        return
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        summary_file = os.path.join(results_dir, 'summary.md')
        with open(summary_file, 'w') as f:
            f.write(f"# Telugu Benchmark Results\n\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Results Summary\n\n")
            f.write("| Task | Metric | Value | Std Error |\n")
            f.write("|------|--------|-------|----------|\n")
            
            for task, task_results in results['results'].items():
                for metric, value in task_results.items():
                    if not metric.endswith('_stderr'):
                        std_err = task_results.get(f"{metric}_stderr", "N/A")
                        f.write(f"| {task} | {metric} | {value:.4f} | {std_err} |\n")
            
            f.write("\n\n## Configuration\n\n")
            f.write("```\n")
            f.write(f"Tasks: {args.tasks}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Few-shot examples: {args.num_fewshot}\n")
            f.write(f"Device: {args.device}\n")
            f.write("```\n")
        
        print(f"Summary generated at {summary_file}")
    
    except Exception as e:
        print(f"Error generating summary: {e}")

if __name__ == "__main__":
    args = parse_arguments()
    results_dir = run_benchmark(args)
    print(f"Benchmark completed. Results saved to {results_dir}")