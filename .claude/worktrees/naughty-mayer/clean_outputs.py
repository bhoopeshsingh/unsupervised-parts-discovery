
import shutil
from pathlib import Path
import os

def clean_outputs():
    """
    Clean up output directories to ensure a fresh run.
    """
    paths_to_clean = [
        Path('parts/clusters'),
        # Path('parts/extracted'), # Don't delete extracted parts unless you want to re-run extraction (slow)
    ]
    
    print("Cleaning output directories...")
    
    for path in paths_to_clean:
        if path.exists():
            print(f"  Removing {path}")
            shutil.rmtree(path)
        else:
            print(f"  {path} does not exist")
            
    # Re-create empty directories
    for path in paths_to_clean:
        path.mkdir(parents=True, exist_ok=True)
        print(f"  Created empty {path}")
        
    print("Cleanup complete!")

if __name__ == "__main__":
    clean_outputs()
