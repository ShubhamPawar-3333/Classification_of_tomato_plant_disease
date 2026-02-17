from setuptools import setup, find_packages

setup(
    name="tomato_disease_advisor",
    version="1.0.0",
    description="Agricultural Disease Diagnosis and Advisory System — Tomato Crops",
    author="Shubham Pawar",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "tensorflow>=2.15.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
        "python-box>=7.0.0",
        "ensure>=1.0.0",
        "gradio>=4.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
    ],
)
