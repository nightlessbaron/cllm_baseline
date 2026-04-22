#!/usr/bin/env python3
# push_to_hub.py — copied verbatim from
# https://gist.github.com/mostafaelhoushi/7c3a2a94ee195491b2e59d1f5126fa18
import argparse
from huggingface_hub import HfApi, HfFolder, upload_folder, snapshot_download
import os
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(
        description="Push a local model folder or a HuggingFace model to the Hugging Face Hub."
    )
    parser.add_argument(
        "--model",
        "--model_dir",
        type=str,
        required=True,
        dest="model",
        help="Path to local model directory OR HuggingFace model ID (e.g. 'Qwen/Qwen2-7B').",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Target repository ID on Hugging Face Hub (e.g. 'username/my-awesome-model').",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token. If not provided, will use token from `huggingface-cli login`.",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload model to Hugging Face Hub",
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="If set, creates the repository as private (if it does not exist).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for downloading HuggingFace models.",
    )

    args = parser.parse_args()

    # Determine if model is a local directory or HuggingFace model ID
    if os.path.isdir(args.model):
        # Local directory
        model_dir = args.model
        print(f"Using local directory: {model_dir}")
    else:
        # Assume it's a HuggingFace model ID - download it
        print(f"Downloading model from HuggingFace: {args.model}")
        try:
            model_dir = snapshot_download(
                repo_id=args.model,
                cache_dir=args.cache_dir,
                token=args.token,
            )
            print(f"Downloaded to: {model_dir}")
        except Exception as e:
            print(f"Error downloading model '{args.model}': {e}")
            sys.exit(1)

    # Validate directory exists
    if not os.path.isdir(model_dir):
        print(f"Error: Directory not found: {model_dir}")
        sys.exit(1)

    # Authenticate
    if args.token:
        HfFolder.save_token(args.token)
    else:
        token = HfFolder.get_token()
        if token is None:
            print("No token found. Please run `huggingface-cli login` or pass --token.")
            sys.exit(1)

    api = HfApi()

    # Create repo if not exists
    print(f"Checking if repo '{args.repo_id}' exists...")
    try:
        api.repo_info(repo_id=args.repo_id, token=args.token)
        print("Repository exists.")
    except Exception:
        print(f"Creating repository '{args.repo_id}'...")
        api.create_repo(repo_id=args.repo_id, token=args.token, private=args.private, exist_ok=True)

    # Upload the folder
    print(f"Uploading from: {model_dir}")
    upload_folder(
        folder_path=model_dir,
        repo_id=args.repo_id,
        commit_message=args.commit_message,
        token=args.token,
    )

    print(f"Upload complete: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
