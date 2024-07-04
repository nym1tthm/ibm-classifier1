import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'ph':7.160476, 'Hardness':179.8898, 'Solids':22003.76, 'Chloramines':5.87086, 'Sulfate':349.2693, 'Conductivity':501.1828 , 'Organic_carbon':17.29771 , 'Trihalomethanes':50.99301 , 'Turbidity':3.626364 })

print(r.json())