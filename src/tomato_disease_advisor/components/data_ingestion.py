"""
Data Ingestion Component

Downloads and extracts the PlantVillage tomato disease dataset.
"""
import os
import zipfile
import urllib.request
from pathlib import Path
from tomato_disease_advisor.entity import DataIngestionConfig


class DataIngestion:
    """
    Handles downloading and extracting the dataset.
    """
    
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize DataIngestion.
        
        Args:
            config: DataIngestionConfig with paths and URLs
        """
        self.config = config
    
    def download_file(self) -> Path:
        """
        Download the dataset file if it doesn't exist.
        
        Returns:
            Path: Path to the downloaded file
        """
        if not os.path.exists(self.config.local_data_file):
            print(f"Downloading dataset from {self.config.source_URL}")
            
            # Create directory if needed
            os.makedirs(self.config.root_dir, exist_ok=True)
            
            # Download with progress
            filename, headers = urllib.request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            
            print(f"Downloaded: {filename}")
            print(f"Headers: {headers}")
        else:
            print(f"File already exists: {self.config.local_data_file}")
        
        return self.config.local_data_file
    
    def extract_zip_file(self) -> Path:
        """
        Extract the downloaded zip file.
        
        Returns:
            Path: Path to the extracted directory
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        print(f"Extracting {self.config.local_data_file} to {unzip_path}")
        
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        
        print(f"Extraction complete!")
        
        # List extracted contents
        contents = os.listdir(unzip_path)
        print(f"Extracted contents: {contents}")
        
        return unzip_path
    
    def run(self) -> Path:
        """
        Execute the complete data ingestion pipeline.
        
        Returns:
            Path: Path to the extracted dataset
        """
        print("=" * 50)
        print("Starting Data Ingestion")
        print("=" * 50)
        
        # Download
        self.download_file()
        
        # Extract
        dataset_path = self.extract_zip_file()
        
        print("=" * 50)
        print("Data Ingestion Complete!")
        print("=" * 50)
        
        return dataset_path
