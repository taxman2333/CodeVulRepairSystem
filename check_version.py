import importlib.metadata

# 需要检查的库列表
required_packages = [
    "transformers",
    "torch",
    "numpy",
    "tqdm",
    "pandas",
    "tokenizers",
    "datasets",
    "gdown",
    "tensorboard",
    "scikit-learn"
]

with open("requirements.txt", "w") as f:
    for package in required_packages:
        try:
            version = importlib.metadata.version(package)
            f.write(f"{package}=={version}\n")
        except importlib.metadata.PackageNotFoundError:
            print(f"[WARNING] {package} is not installed")

print("Generated requirements.txt")