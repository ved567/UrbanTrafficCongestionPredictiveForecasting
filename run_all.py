import subprocess
import time
import sys
import os
import signal

def check_env():
    """
    Verifies that the .env file exists in the root directory.
    If missing, terminates the process with a critical error message.
    """
    if not os.path.exists(".env"):
        print("CRITICAL ERROR: .env file not found!")
        print("Please create a .env file based on .env.example with your TOMTOM_API_KEY.")
        sys.exit(1)

def run_pipeline():
    """
    Orchestrates the end-to-end traffic prediction pipeline.
    Manages background processes and executes data ingestion, 
    feature engineering, and model training steps sequentially.
    """
    check_env()
    
    print("--- Starting Urban Traffic Predictive System ---")

    print("Step 1: Launching Background Traffic Collector...")
    
    collector_process = subprocess.Popen(
        [sys.executable, "scripts/traffic_collector.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    
    terminated = False

    def shutdown_handler(signum, frame):
        """
        Handles graceful shutdown of the background collector process 
        upon receiving termination signals.
        """
        nonlocal terminated
        if not terminated:
            print("\n\n--- Shutdown Signal Received ---")
            print(f"Stopping Background Collector (PID: {collector_process.pid})...")
            collector_process.terminate()
            try:
                collector_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Collector did not stop gracefully, forcing kill...")
                collector_process.kill()
            print("System Shutdown Cleanly.")
            terminated = True
            sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    time.sleep(5)

    try:
        print("Step 2: Seeding Historical Data...")
        subprocess.run([sys.executable, "scripts/db_seeder.py"], check=True)

        print("Step 3: Processing Accident Data (NoSQL)...")
        subprocess.run([sys.executable, "scripts/accident_ingestion.py"], check=True)

        print("Step 4: Merging SQL + NoSQL + Weather...")
        subprocess.run([sys.executable, "scripts/feature_engineering.py"], check=True)

        print("Step 5: Training LSTM Model...")
        subprocess.run([sys.executable, "models/model_training.py"], check=True)

        print("Step 6: Evaluating Models (LSTM vs ARIMA)...")
        subprocess.run([sys.executable, "models/evaluate_models.py", "--epochs", "50"], check=True)

        print("\n--- Pipeline Execution Complete ---")
        print("The Traffic Collector is still running in the background.")
        print("You can now launch the dashboard: streamlit run dashboard/app.py")
        print("Press Ctrl+C to stop the collector and exit.")
        
        while True:
            time.sleep(1)

    except subprocess.CalledProcessError as e:
        print(f"\nPipeline Failed at step: {e.cmd}")
        shutdown_handler(None, None)
    except Exception as e:
        if not isinstance(e, SystemExit):
            print(f"\nAn unexpected error occurred: {e}")
            shutdown_handler(None, None)

if __name__ == "__main__":
    run_pipeline()
