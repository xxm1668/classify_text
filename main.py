from flask import Flask, request
from predict import predict
import re
import traceback

app = Flask(__name__)

pattern_en = re.compile("[A-Za-z]{6,}")


@app.route("/classify", methods=['POST'])
def text_extract():
    text = request.json.get("text").strip()
    print(text)
    data = ''
    code = ''
    try:
        data = predict(text)
        code = '200'
    except:
        code = '404'
        print(traceback.format_exc())
    finally:
        return {'code': code, 'data': data}


if __name__ == '__main__':
    app.run(port=8011, debug=True, host='172.16.19.81')  # 启动服务
