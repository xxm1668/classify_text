import re
import os
import json
import requests

key = {
    0: '住宿服务',
    1: '餐饮服务',
    2: '道路旅客运输服务',
    3: '基础电信服务',
    4: '汽油',
    5: '交通运输服务'
}
url = 'http://172.16.19.81:8011/classify'
url2 = 'http://172.16.19.81:8050/classify'
target_filename = r'/Users/haojingkun/Downloads/Textclassification/ceshi.txt'
target_w = open(target_filename, 'a+', encoding='utf-8')
filename = r'/Users/haojingkun/Downloads/Textclassification/test.txt'
with open(filename, 'r', encoding='utf-8') as data:
    lines = data.readlines()
    for line in lines:
        line = line.strip()
        texts = line.split('\t')
        data2 = {'text': texts[0]}
        response = requests.post(url, json=data2)
        json_data3 = str(response.content, encoding='utf-8')
        json_data3 = json.loads(json_data3)
        label = json_data3['data']
        content = texts[0].split(',')[0]
        data2 = {'text': content}
        response2 = requests.post(url2, json=data2)
        json_data = str(response2.content, encoding='utf-8')
        json_data = json.loads(json_data)
        try:
            label2 = json_data['ret{}'][0]
            label2 = label2['L0_category']
        except:
            continue
        print('-----')
        target_w.write(texts[0] + '\t' + label + '\t' + label2 + '\n')
