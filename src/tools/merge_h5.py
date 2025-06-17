"""Functions that combines embeddings across JUMPCP plates"""
import os
import argparse
from typing import Tuple, List

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Merge multiple H5 files containing embeddings.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input H5 files",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output H5 file path. If not provided, will use 'merged.h5' in input directory",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="BR",
        help="Prefix for filtering input files (default: 'BR')",
    )
    return parser.parse_args()


def get_input_files(input_dir: str, prefix: str) -> List[str]:
    """Get sorted list of input files matching the prefix."""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    d_names = [f for f in os.listdir(input_dir) if f.startswith(prefix)]
    d_names.sort()
    return d_names


def process_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Process a single H5 file and return processed names and embeddings."""
    plate = os.path.basename(file_path).split("_")[0]
    
    with h5py.File(file_path, "r") as f:
        names = f["names"][:]
        embeddings = f["embeddings"][:]

        # Update names with the plate number prefix
        updated_names = [
            f"{plate}-{name.decode('utf-8').split('-')[1]}-{name.decode('utf-8').split('-')[0]}"
            if not name.decode("utf-8").startswith("BR")
            else f"{name.decode('utf-8').split('-')[0]}-{name.decode('utf-8').split('-')[2]}-{name.decode('utf-8').split('-')[1]}"
            for name in names
        ]
        updated_names = np.array([n.encode("utf-8") for n in updated_names], dtype="S")
        
    return updated_names, embeddings


def main():
    """Main function to merge H5 files."""
    args = parse_args()
    
    # Set output path if not provided
    output_path = args.output_file or os.path.join(args.input_dir, "merged.h5")
    
    try:
        # Get input files
        input_files = get_input_files(args.input_dir, args.prefix)
        if not input_files:
            raise ValueError(f"No files found with prefix '{args.prefix}' in {args.input_dir}")

        # Process all files
        all_names = []
        all_embeddings = []
        
        for file in input_files:
            file_path = os.path.join(args.input_dir, file)
            names, embeddings = process_file(file_path)
            all_names.append(names)
            all_embeddings.append(embeddings)

        # Concatenate results
        all_names = np.concatenate(all_names, axis=0)
        all_embeddings = np.concatenate(all_embeddings, axis=0)

        # Write output
        with h5py.File(output_path, "w") as combined_file:
            combined_file.create_dataset("names", data=all_names)
            combined_file.create_dataset("embeddings", data=all_embeddings)
            
        print(f"Successfully merged {len(input_files)} files to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
