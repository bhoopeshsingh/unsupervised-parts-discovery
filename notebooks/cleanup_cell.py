# Add this cell at the beginning of your notebook to clean up output folders
import shutil
from pathlib import Path
import os

def clean_outputs():
    """
    Clean up output directories to ensure a fresh run.
    """
    print("Cleaning output directories...")
    
    # robustly find project root
    # If running as script
    if '__file__' in globals():
        current_file = Path(__file__).resolve()
        # If script is in notebooks/, root is parent
        if current_file.parent.name == 'notebooks':
            project_root = current_file.parent.parent
        else:
            # Fallback or assume CWD if running from root
            project_root = Path.cwd()
    else:
        # If running in notebook (jupyter)
        # Assuming notebook is in notebooks/ folder
        project_root = Path('..').resolve()
        
    print(f"Project root detected as: {project_root}")
    
    # Define paths to clean (relative to project root)
    paths_to_clean = [
        project_root / 'parts/clusters',
        # project_root / 'parts/extracted', # Uncomment if you want to re-extract features
    ]
    
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

# Run cleanup
if __name__ == "__main__":
    clean_outputs()
