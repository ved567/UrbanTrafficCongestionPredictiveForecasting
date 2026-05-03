import requests
import pandas as pd


# Your new TomTom ID Badge
API_KEY = "Bs4uffOtSpdO8Fj7QF2legoU1iRowti8"

def fetch_tomtom_flow():
    # Point on FDR Drive (42nd St)
    point = "40.7478,-73.9718"
    
    # TomTom Flow Segment API URL
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key={API_KEY}&point={point}"
    
    print("📥 Connecting to TomTom Servers...")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        # TomTom gives us a nice dictionary of flow data
        flow = data['flowSegmentData']
        
        # Create a tiny dataframe to see the result
        df = pd.DataFrame([{
            'currentSpeed': flow['currentSpeed'],
            'freeFlowSpeed': flow['freeFlowSpeed'],
            'confidence': flow['confidence'],
            'timestamp': pd.Timestamp.now()
        }])
        
        print("✅ SUCCESS! Real-time speed from TomTom:")
        print(df)
        df.to_csv("tomtom_fdr_flow.csv", index=False)
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    fetch_tomtom_flow()
