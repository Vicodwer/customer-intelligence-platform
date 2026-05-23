from pathlib import Path


def test_required_project_folders_exist():
    required_dirs = [
        "data",
        "data/raw",
        "data/processed",
        "data/samples",
        "src",
        "src/data_pipeline",
        "src/training",
        "src/serving",
        "src/rag",
        "tests",
        "pipelines",
        "app",
        "monitoring",
        "docs",
    ]

    for directory in required_dirs:
        assert Path(directory).exists(), f"Missing required directory: {directory}"


def test_required_entrypoint_files_exist():
    required_files = [
        "src/data_pipeline/ingest.py",
        "src/data_pipeline/validate.py",
        "src/data_pipeline/features.py",
        "src/training/train.py",
        "src/training/evaluate.py",
        "src/serving/serve.py",
        "src/rag/build_index.py",
        "src/rag/retrieve.py",
        "src/rag/answer.py",
        "src/rag/rag_eval.py",
        "README.md",
        "requirements.txt",
    ]

    for file_path in required_files:
        assert Path(file_path).exists(), f"Missing required file: {file_path}"