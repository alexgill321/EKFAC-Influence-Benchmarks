from huggingface_hub import snapshot_download
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="/scratch/general/u1380656/pile")

args = parser.parse_args()

snapshot_download(
    repo_id="EleutherAI/pile-standard-pythia-preshuffled", 
    repo_type="dataset", 
    cache_dir=args.output_dir
)