import os
import zipfile
import random
import shutil
from google.colab import drive

def extract_sample_images(zip_path, output_dir, sample_size=(20000,20000)):
    """
    Extract a random sample of images from a zip file
    
    Parameters:
    zip_path: Path to zip file on Google Drive
    output_dir: Directory to save sampled images
    sample_size: Tuple of (min, max) number of images to extract
    """
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all images from zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get all file names from zip
        all_files = [f for f in zip_ref.namelist() if 
                    f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        # Determine sample size
        total_images = len(all_files)
        sample_size = min(random.randint(sample_size[0], sample_size[1]), total_images)
        
        print(f"Total images in zip: {total_images}")
        print(f"Extracting {sample_size} images...")
        
        # Randomly sample files
        selected_files = random.sample(all_files, sample_size)
        
        # Extract selected files
        for i, file_name in enumerate(selected_files):
            zip_ref.extract(file_name, output_dir)
            if (i + 1) % 1000 == 0:
                print(f"Extracted {i + 1} images...")

    print("Extraction complete!")
    return len(selected_files)

# Example usage
zip_path = '/content/drive/MyDrive/img_align_celeba.zip'  
output_dir = '/content/Generative_Image_Using_VAE/data/raw1'

# Extract images
num_extracted = extract_sample_images(
    zip_path=zip_path,
    output_dir=output_dir,
    sample_size=(20000,20000)
)

print(f"Successfully extracted {num_extracted} images to {output_dir}")