from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="EleutherAI/pile-standard-pythia-preshuffled", 
    repo_type="dataset", 
    cache_dir="/scratch/general/u1380656/pile"
)