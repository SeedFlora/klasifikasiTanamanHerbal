"""
Deploy Indonesian Herbal Plants Classifier to Hugging Face Spaces
"""
import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
import argparse

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def deploy_to_huggingface(username: str, token: str, space_name: str = "indonesian-herbal-classifier"):
    """Deploy Gradio app to Hugging Face Spaces"""

    print("Deploying to Hugging Face Spaces...")
    print(f"   Space: {username}/{space_name}")

    # Initialize API
    api = HfApi()
    repo_id = f"{username}/{space_name}"

    # Step 1: Create Space
    print("\nStep 1: Creating Space...")
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="space",
            space_sdk="gradio",
            private=False
        )
        print(f"   Space created: {repo_id}")
    except Exception as e:
        print(f"   Space might already exist: {e}")

    # Step 2: Prepare files
    print("\nStep 2: Preparing files...")
    temp_dir = Path("temp_hf_deploy")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Copy essential files
        files_to_copy = [
            ("app.py", "app.py"),
            ("requirements.txt", "requirements.txt"),
            ("README_HF.md", "README.md"),
        ]

        for src, dst in files_to_copy:
            if Path(src).exists():
                shutil.copy(src, temp_dir / dst)
                print(f"    Copied {src} → {dst}")

        # Copy src directory
        shutil.copytree("src", temp_dir / "src", dirs_exist_ok=True)
        print(f"    Copied src/")

        # Copy outputs directory (models & results)
        output_dest = temp_dir / "outputs"
        output_dest.mkdir(exist_ok=True)

        # Copy model files (only best models to save space)
        models_to_copy = ["efficientnetv2.pth", "yolov11.pth"]  # Best 2 models
        models_dir = Path("outputs/models")
        models_dest = output_dest / "models"
        models_dest.mkdir(exist_ok=True)

        for model_file in models_to_copy:
            src_path = models_dir / model_file
            if src_path.exists():
                shutil.copy(src_path, models_dest / model_file)
                size_mb = src_path.stat().st_size / (1024 * 1024)
                print(f"    Copied {model_file} ({size_mb:.1f} MB)")

        # Copy results and class names
        if Path("outputs/training_results.json").exists():
            shutil.copy("outputs/training_results.json", output_dest / "training_results.json")
            print(f"    Copied training_results.json")

        if Path("outputs/class_names.json").exists():
            shutil.copy("outputs/class_names.json", output_dest / "class_names.json")
            print(f"    Copied class_names.json")

        # Step 3: Upload to HF
        print(f"\n Step 3: Uploading to Hugging Face...")
        upload_folder(
            folder_path=str(temp_dir),
            repo_id=repo_id,
            repo_type="space",
            token=token,
            commit_message="Deploy Indonesian Herbal Plants Classifier"
        )

        print(f"\n Deployment successful!")
        print(f" Your app is available at:")
        print(f"   https://huggingface.co/spaces/{repo_id}")

    except Exception as e:
        print(f"\n Error during deployment: {e}")
        raise

    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\n Cleaned up temporary files")

def main():
    parser = argparse.ArgumentParser(description="Deploy to Hugging Face Spaces")
    parser.add_argument("--username", required=True, help="Your HuggingFace username")
    parser.add_argument("--token", required=True, help="Your HuggingFace token")
    parser.add_argument("--space-name", default="indonesian-herbal-classifier",
                       help="Space name (default: indonesian-herbal-classifier)")

    args = parser.parse_args()

    deploy_to_huggingface(args.username, args.token, args.space_name)

if __name__ == "__main__":
    # You can also run directly with:
    # python deploy_to_hf.py --username YOUR_USERNAME --token YOUR_TOKEN

    # Or uncomment and fill these:
    # HF_USERNAME = "YOUR_USERNAME"
    # HF_TOKEN = "YOUR_TOKEN"
    # deploy_to_huggingface(HF_USERNAME, HF_TOKEN)

    main()
