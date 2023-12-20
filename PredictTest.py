"""
This notebook is designed to test the predict.py script
"""
import requests
url = 'http://localhost:9696/predict'

# Cell JSON to input
penguin = {
    "bill_length_mm": 20, 
    "bill_depth_mm": 10,
    "flipper_length_mm": 100, 
    "body_mass_g": 500.0, 
    "sex": "female", 
    "diet": "squid",
    "life_stage": "adult",
    "health_metrics": "overweight"
}

response = requests.post(url, json=penguin).json()

print("Species: ", response['Species'])
print("Probability: ", response['Probability'])