from huggingface_hub import HfApi
import os
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="./dataset_storage/baby-lm-strict.hf",
    repo_id="InoWouw/BabyLM-strict",
    repo_type="dataset",
)
