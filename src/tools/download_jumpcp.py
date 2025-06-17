import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the S3 base URL and local base path
s3_base_url = "s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/images/"
local_base_path = "/data/nobackup/mingyulu/datasets/2020_11_04_CPJUMP1/images"  

# Function to download a single folder
def download_folder(s3_folder, local_folder):
    command = [
        "aws", "s3", "cp", "--no-sign-request", "--recursive",
        f"{s3_base_url}{s3_folder}", f"{local_base_path}/{s3_folder}"
    ]
    
    subprocess.run(command, check=True)
    print(f"Downloaded {s3_folder}")

# List all subfolders in the directory
def list_subfolders():
    command = ["aws", "s3", "ls", s3_base_url, "--no-sign-request"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    folders = [
        line.split()[-1] for line in result.stdout.splitlines() if line.endswith("/")
    ]

    return folders

# Main function to download folders in parallel
def parallel_download():
    subfolders = list_subfolders()
    with ThreadPoolExecutor(max_workers=32) as executor:  # Adjust the number of workers as needed
        future_to_folder = {executor.submit(download_folder, folder, local_base_path): folder for folder in subfolders}
        for future in as_completed(future_to_folder):
            folder = future_to_folder[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error downloading {folder}: {e}")

if __name__ == "__main__":
    parallel_download()
