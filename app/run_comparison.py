import streamlit as st
import subprocess
import os
import sys
import argparse
import shutil

def create_test_data(test_dir):
    """Create a test data directory with sample images for testing the comparison tool"""
    print(f"Creating test data directory: {test_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(test_dir, exist_ok=True)
    
    # Check if we have images in data or old_data that we can copy
    source_dirs = ["data", "old_data"]
    found_files = False
    
    for source_dir in source_dirs:
        source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), source_dir)
        if os.path.exists(source_path):
            # Look for image files
            image_files = []
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                for f in os.listdir(source_path):
                    if f.lower().endswith(ext) and 'label' not in f:
                        image_files.append(os.path.join(source_path, f))
            
            if image_files:
                found_files = True
                # Copy at most 3 files to test directory
                for i, f in enumerate(image_files[:3]):
                    # Create different versions to simulate different resolutions
                    base_name = os.path.basename(f)
                    name, ext = os.path.splitext(base_name)
                    
                    # Copy original
                    shutil.copy(f, os.path.join(test_dir, f"{name}_original{ext}"))
                    print(f"Created test file: {name}_original{ext}")
                    
                    # Create a "4x" version
                    shutil.copy(f, os.path.join(test_dir, f"{name}_4x{ext}"))
                    print(f"Created test file: {name}_4x{ext}")
                    
                    # Create a "10x" version
                    shutil.copy(f, os.path.join(test_dir, f"{name}_10x{ext}"))
                    print(f"Created test file: {name}_10x{ext}")
                
                # We only need one source directory
                break
    
    if not found_files:
        print("No source images found in data or old_data directories.")
        return False
    
    return True

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the multi-resolution comparison tool")
    parser.add_argument("--create-test-data", action="store_true", help="Create a test_data directory with sample images")
    args = parser.parse_args()
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the working directory to the script directory
    os.chdir(current_dir)
    
    # Create test data if requested
    if args.create_test_data:
        test_dir = os.path.join(current_dir, "test_data")
        if create_test_data(test_dir):
            print(f"Test data created in {test_dir}")
            print("You can now run the comparison tool with the test data.")
        else:
            print("Failed to create test data.")
        sys.exit(0)
    
    print("Starting the multi-resolution comparison tool...")
    
    # Run the comparison tool with Streamlit
    try:
        subprocess.run(["streamlit", "run", "compare_resolutions.py"], check=True)
    except KeyboardInterrupt:
        print("\nExiting the comparison tool...")
    except Exception as e:
        print(f"Error running the comparison tool: {e}") 