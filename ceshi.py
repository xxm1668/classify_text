import requests
import json

url = 'http://172.16.19.81:8030/predict'
content = ''
data = {'text': content}
response2 = requests.post(url, json=data)
json_data = str(response2.content, encoding='utf-8')
json_data = json.loads(json_data)
print(json_data)
