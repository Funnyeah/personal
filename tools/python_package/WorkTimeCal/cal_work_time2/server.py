#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : server.py
# @Author: zhiyong Teng
# @Date  : 2023/8/25
# @Desc  :
import gradio as gra
from get_work_time import cal_work_time
def main(token):
    res = cal_work_time(token)
    return str(res)

app =  gra.Interface(fn = main, inputs="text", outputs="text")
#run the app
# app.launch()
app.launch(server_name="0.0.0.0", server_port=7860, share=True)

# app.queue().launch(share=False,
#                         debug=False,
#                          server_name="0.0.0.0",
#                          server_port=8433,
#                          ssl_verify=False,
#                          ssl_certfile="cert.pem",
#                          ssl_keyfile="key.pem")

