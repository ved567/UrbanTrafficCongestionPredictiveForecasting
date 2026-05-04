import subprocess
import time
import sys
import os

def run_pipeline():
    print("--- Starting Urban Traffic Predictive System ---")

    # 1. Start the Traffic Collector as a background process
    print("Step 1: Launching Background Traffic Collector...")
    collector_process = subprocess.Popen(
        [sys.executable, "scripts/traffic_collector.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    
    # Give the collector a few seconds to initialize the database
    time.sleep(5)

    try:
        # 2. Run Step 1: NoSQL Accident Processing
        print("Step 2: Processing Accident Data (NoSQL)...")
        subprocess.run([sys.executable, "scripts/step1_mongo.py"], check=True)

        # 3. Run Step 2: Data Merging
        print("Step 3: Merging SQL + NoSQL + Weather...")
        subprocess.run([sys.executable, "scripts/step2_merge.py"], check=True)

        # 4. Run Step 3: AI Model Training
        print("Step 4: Training LSTM Model...")
        subprocess.run([sys.executable, "models/step3_train.py"], check=True)

        print("\n--- Pipeline Execution Complete ---")
        print("The Traffic Collector is still running in the background.")
        print("Press Ctrl+C to stop the collector and exit.")
        
        # Keep the script alive so the background collector can keep working
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping Background Collector...")
        collector_process.terminate()
        print("System Shutdown.")
    except subprocess.CalledProcessError as e:
        print(f"\nPipeline Failed at step: {e.cmd}")
        collector_process.terminate()

if __name__ == "__main__":
    run_pipeline()