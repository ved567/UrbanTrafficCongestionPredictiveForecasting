import requests

import pandas as pd



API_KEY = "Bs4uffOtSpdO8Fj7QF2legoU1iRowti8"



def fetch_tomtom_flow():

    point = "40.7478,-73.9718"

    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key={API_KEY}&point={point}"

    

    response = requests.get(url)

    

    if response.status_code == 200:

        data = response.json()

        flow = data['flowSegmentData']

        

        df = pd.DataFrame([{

            'currentSpeed': flow['currentSpeed'],

            'freeFlowSpeed': flow['freeFlowSpeed'],

            'confidence': flow['confidence'],

            'timestamp': pd.Timestamp.now()

        }])

        

        df.to_csv("tomtom_fdr_flow.csv", index=False)

    else:

        pass



if __name__ == "__main__":

    fetch_tomtom_flow()
