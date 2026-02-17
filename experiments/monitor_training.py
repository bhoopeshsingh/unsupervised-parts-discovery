
import sys
import mlflow
from mlflow.tracking import MlflowClient
import time
import os

# Configuration from unified_config.yaml (hardcoded or loaded)
TRACKING_URI = "http://0.0.0.0:5050"
EXPERIMENT_NAME = "interpretable_space"

def check_slots():
    print(f"Connecting to MLflow at {TRACKING_URI}...")
    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        client = MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        
        if experiment is None:
            print(f"Experiment '{EXPERIMENT_NAME}' not found.")
            return

        # Get latest run
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attribute.start_time DESC"],
            max_results=1
        )
        
        if not runs:
            print("No runs found.")
            return

        latest_run = runs[0]
        run_id = latest_run.info.run_id
        status = latest_run.info.status
        
        print(f"Latest Run ID: {run_id} ({status})")
        
        # Get metrics history for active slots
        metric_history = client.get_metric_history(run_id, "train_active_slots")
        if metric_history:
            latest_val = metric_history[-1].value
            print(f"Current Active Slots: {latest_val:.2f}")
            
            # Check for collapse
            if latest_val < 3.0:
                 print("⚠️ WARNING: Slot collapse detected! (Active slots < 3)")
            else:
                 print("✅ Slots appear stable.")
                 
            # Print history
            print("History (last 5 steps):")
            for m in metric_history[-5:]:
                print(f"  Step {m.step}: {m.value:.2f}")
        else:
            print("Metric 'train_active_slots' not found yet.")
            
        # Check other metrics
        loss_hist = client.get_metric_history(run_id, "train_total_loss")
        if loss_hist:
            print(f"Current Loss: {loss_hist[-1].value:.4f}")

    except Exception as e:
        print(f"Error checking MLflow: {e}")
        # Fallback: maybe tracking URI is file:./mlruns?
        if TRACKING_URI != "file:./mlruns" and not os.path.exists("./mlruns"):
             print("Try running `mlflow ui` to start the server if using http.")

if __name__ == "__main__":
    check_slots()
