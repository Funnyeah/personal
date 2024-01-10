from flask import Flask, request
from get_work_time import cal_work_time
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # 防止jsonify返回中文乱码

@app.route('/cal_work_time')
def hello():
#     token = request.args.get("token")
#     token = request.cookies
#     print(token)
    
    work_days, hours = cal_work_time(None)
    return {"work_days": work_days, "hours": hours}

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True, port=6789)

