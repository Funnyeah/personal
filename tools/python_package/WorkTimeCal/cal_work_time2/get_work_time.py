import re
import requests
import numpy as np
from datetime import datetime, timedelta


def get_week_start_and_end_dates(year, week_number):
    start_date = datetime.strptime(f"{year}-W{week_number-1}-1", "%Y-W%U-%w")
    end_date = start_date + timedelta(days=6)
    return start_date.date(), end_date.date()



# print(f"当前日期：{current_date.date()}")
# print(f"当前年份：{current_year}")
# print(f"当前周数：{current_week + 1}")  # 周数从0开始，所以需要加1
# print(f"当前周的起始日期：{start_date_of_current_week}")


def get_current_week_range():
    today = datetime.today()
    day_of_week = today.weekday()
    start_of_week = today - timedelta(days=day_of_week)
    end_of_week = start_of_week + timedelta(days=6)
    return start_of_week.date(), end_of_week.date()

def get_dates_in_range(start_date, end_date):
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    return dates



def calculate_work_duration(start_time, end_time):
    if start_time is None or end_time is None:
        return None
    start_datetime = datetime.strptime(start_time, '%Y-%m-%d %H:%M')
    end_datetime = datetime.strptime(end_time, '%Y-%m-%d %H:%M')
    duration = end_datetime - start_datetime
    return  duration.seconds


def request_(data, token, current_year, current_month):
    url = f"http://ehr.songguo7.com/psc/HR92PRD/EMPLOYEE/HRMS/s/WEBLIB_FP_PORTA.FP_ABSENCE_IS.FieldFormula.IScript_AbsenceCalendarData?year={current_year}&month={current_month}&emplid="

    # token = "_ga=GA1.2.104219100.1635756921; cna=XL3/GaeJzDwCAXt/Nywrh+vQ; ExpirePage=http://ehr.songguo7.com/psc/HR92PRD/; PS_LOGINLIST=http://ehr.songguo7.com/HR92PRD; PS_DEVICEFEATURES=new:1; SignOnDefault=; PS_LASTSITE=http://ehr.songguo7.com/psc/HR92PRD/; ehrweb3-8000-PORTAL-PSJSESSIONID=Nc7-WakQSY9CXH6wG5lUIPzgoaHB1b3I!1347812039; Fu-Xi-Token=eyJhbGciOiJIUzI1NiJ9.eyJnbG9iYWxfdXNlcl9pZCI6MTcxNTcsImdsb2JhbF91c2VyX25hbWUiOiLmu5Xlv5fli4ciLCJleHAiOjE2OTI1ODk0NzV9.SZm0hLoILXdc6Rrgd2TtwvzWtD79SXIYyrr0v6Xg1W8; PS_TokenSite=http://ehr.songguo7.com/psc/HR92PRD/?ehrweb1-8000-PORTAL-PSJSESSIONID; access_token=eyJhbGciOiJIUzI1NiJ9.eyJnbG9iYWxfdXNlcl9pZCI6MTcxNTcsImdsb2JhbF91c2VyX25hbWUiOiLmu5Xlv5fli4ciLCJleHAiOjE2OTMxMzQ2MzV9.HczLxAWJxHGuz9pwnTcviO1cl2w340HORuXwWwrZbvE; refresh_token=C48B04CB40464462848D1A56EDE2DF80; ehrweb1-8000-PORTAL-PSJSESSIONID=e5onPjQp0jUAbfrNKyuhvvI1SEM2iSK9!-1896598640; PS_TOKENEXPIRE=24_Aug_2023_11:11:42_GMT; PS_TOKEN=qwAAAAQDAgEBAAAAvAIAAAAAAAAsAAAABABTaGRyAk4Adwg4AC4AMQAwABQL/N9X3VyZqZ+TAwl1zhjmvO5YN2sAAAAFAFNkYXRhX3icLYo7DkBQAATHJ0qlWxCeV3ABUQoqjUrQ0EiEszmcjdhiZjfZE/A913Hkx+VLdDCxMXOzsHKxfysYqOkIG6GiZ9RssYYUQ04sF6LBipmciJlo/271MvqU8AKeiQ6M"

    # token = "_ga=GA1.2.104219100.1635756921; cna=XL3/GaeJzDwCAXt/Nywrh+vQ; PS_LOGINLIST=http://ehr.songguo7.com/HR92PRD; PS_DEVICEFEATURES=new:1; SignOnDefault=; ehrweb1-8000-PORTAL-PSJSESSIONID=e5onPjQp0jUAbfrNKyuhvvI1SEM2iSK9!-1896598640; ExpirePage=http://ehr.songguo7.com/psp/HR92PRD/; ps_theme=node:HRMS portal:EMPLOYEE theme_id:C_PR_STYLE accessibility:N formfactor:0 piamode:3; HPTabName=DEFAULT; HPTabNameRemote=; LastActiveTab=DEFAULT; PS_DEVICEFEATURES=width:2560 height:1440 pixelratio:1.25 touch:0 geolocation:1 websockets:1 webworkers:1 datepicker:1 dtpicker:1 timepicker:1 dnd:1 sessionstorage:1 localstorage:1 history:1 canvas:1 svg:1 postmessage:1 hc:0 maf:0; psback=%22%22url%22%3A%22http%3A%2F%2Fehr.songguo7.com%2Fpsp%2FHR92PRD%2FEMPLOYEE%2FHRMS%2Fh%2F%3Ftab%3DDEFAULT%22%20%22label%22%3A%22%E4%B8%BB%E9%A1%B5%22%20%22origin%22%3A%22PIA%22%20%22layout%22%3A%220%22%20%22refurl%22%3A%22http%3A%2F%2Fehr.songguo7.com%2Fpsp%2FHR92PRD%2FEMPLOYEE%2FHRMS%2Fh%2F%3Ftab%3DDEFAULT%22%22; Fu-Xi-Token=eyJhbGciOiJIUzI1NiJ9.eyJnbG9iYWxfdXNlcl9pZCI6MTcxNTcsImdsb2JhbF91c2VyX25hbWUiOiLmu5Xlv5fli4ciLCJleHAiOjE2OTMyMzAwMjB9.yXkxptwcuWjWJMukE1m0RqcqtvF1XKUNKpL4jgqj-TU; access_token=eyJhbGciOiJIUzI1NiJ9.eyJnbG9iYWxfdXNlcl9pZCI6MTcxNTcsImdsb2JhbF91c2VyX25hbWUiOiLmu5Xlv5fli4ciLCJleHAiOjE2OTM0NTAwMTh9.iS9pTc9Y3WssFmIfy_UIGAewpP-XdvA-zqntro-KLs4; refresh_token=D43FBAB4B26047E6BCA3926764C17301; ehrweb3-8000-PORTAL-PSJSESSIONID=pKA_HvaKBTS4PfybHNof2jKWHCJfvSCo!1347812039; PS_TOKENEXPIRE=29_Aug_2023_02:28:28_GMT; PS_TOKEN=qQAAAAQDAgEBAAAAvAIAAAAAAAAsAAAABABTaGRyAk4Adwg4AC4AMQAwABSnVqm1jxUUNuah9xGsKKRkByEO72kAAAAFAFNkYXRhXXicPYcxDkBAFAVnEaXSLQi7IusCohRUGpWgoZEIZ3M4P1t4xbyZCwh8Tyn518MtPpnZWXhY2bg5XIUjDT1RK6gZmCQ7Ck2GxpDIW6Gmcm5Ixe3PUpg7N/ABoHkOng==; PS_TokenSite=http://ehr.songguo7.com/psc/HR92PRD/?ehrweb3-8000-PORTAL-PSJSESSIONID; PS_LASTSITE=http://ehr.songguo7.com/psc/HR92PRD/"
    # token = "_ga=GA1.2.104219100.1635756921; cna=XL3/GaeJzDwCAXt/Nywrh+vQ; PS_LOGINLIST=http://ehr.songguo7.com/HR92PRD; PS_DEVICEFEATURES=new:1; SignOnDefault=; ehrweb1-8000-PORTAL-PSJSESSIONID=e5onPjQp0jUAbfrNKyuhvvI1SEM2iSK9!-1896598640; ExpirePage=http://ehr.songguo7.com/psp/HR92PRD/; ps_theme=node:HRMS portal:EMPLOYEE theme_id:C_PR_STYLE accessibility:N formfactor:0 piamode:3; HPTabName=DEFAULT; HPTabNameRemote=; LastActiveTab=DEFAULT; PS_DEVICEFEATURES=width:2560 height:1440 pixelratio:1.25 touch:0 geolocation:1 websockets:1 webworkers:1 datepicker:1 dtpicker:1 timepicker:1 dnd:1 sessionstorage:1 localstorage:1 history:1 canvas:1 svg:1 postmessage:1 hc:0 maf:0; psback=%22%22url%22%3A%22http%3A%2F%2Fehr.songguo7.com%2Fpsp%2FHR92PRD%2FEMPLOYEE%2FHRMS%2Fh%2F%3Ftab%3DDEFAULT%22%20%22label%22%3A%22%E4%B8%BB%E9%A1%B5%22%20%22origin%22%3A%22PIA%22%20%22layout%22%3A%220%22%20%22refurl%22%3A%22http%3A%2F%2Fehr.songguo7.com%2Fpsp%2FHR92PRD%2FEMPLOYEE%2FHRMS%2Fh%2F%3Ftab%3DDEFAULT%22%22; Fu-Xi-Token=eyJhbGciOiJIUzI1NiJ9.eyJnbG9iYWxfdXNlcl9pZCI6MTcxNTcsImdsb2JhbF91c2VyX25hbWUiOiLmu5Xlv5fli4ciLCJleHAiOjE2OTMyMzAwMjB9.yXkxptwcuWjWJMukE1m0RqcqtvF1XKUNKpL4jgqj-TU; ehrweb3-8000-PORTAL-PSJSESSIONID=pKA_HvaKBTS4PfybHNof2jKWHCJfvSCo!1347812039; PS_TOKEN=qQAAAAQDAgEBAAAAvAIAAAAAAAAsAAAABABTaGRyAk4Adwg4AC4AMQAwABSnVqm1jxUUNuah9xGsKKRkByEO72kAAAAFAFNkYXRhXXicPYcxDkBAFAVnEaXSLQi7IusCohRUGpWgoZEIZ3M4P1t4xbyZCwh8Tyn518MtPpnZWXhY2bg5XIUjDT1RK6gZmCQ7Ck2GxpDIW6Gmcm5Ixe3PUpg7N/ABoHkOng==; PS_TokenSite=http://ehr.songguo7.com/psc/HR92PRD/?ehrweb3-8000-PORTAL-PSJSESSIONID; PS_LASTSITE=http://ehr.songguo7.com/psc/HR92PRD/; PS_TOKENEXPIRE=29_Aug_2023_02:30:51_GMT; access_token=eyJhbGciOiJIUzI1NiJ9.eyJnbG9iYWxfdXNlcl9pZCI6MTcxNTcsImdsb2JhbF91c2VyX25hbWUiOiLmu5Xlv5fli4ciLCJleHAiOjE2OTM3MTA4NjF9.VrdECH4T9UQSMwCoRpKQdEzr-WrPHmi5AABIQJ-4f1Q; refresh_token=B56F1448C335487888E5961EA83B93B1"
    headers = {"Cookie": token.strip()}
    print(url)
    text = requests.post(url, headers=headers)
    print(text.text)
    print(text.json())
    card_list = text.json()["calendar"][1]["weeks"][1:]

    pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})"


    for row in card_list:
        for key, val in row.items():
            for day in val:
                if "cardtime-list" not in day:
                    continue
                day_info = {"上班": None, "下班": None}
                for card in day["cardtime-list"]:
                    if "上班" in card["card_time_descr"]:
                        come_time = re.findall(pattern, card["card_time_descr"])
                        day_info["上班"] = come_time[0]

                    if "下班" in card["card_time_descr"]:
                        out_time = re.findall(pattern, card["card_time_descr"])
                        day_info["下班"] = out_time[0]
                if day_info["上班"]:
                    date = day_info["上班"].split(" ")[0]
                    if date in data:
                        data[date] = day_info

def get_last_and_current_months():
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month

    last_month_date = current_date - timedelta(days=current_date.day)
    last_month_year = last_month_date.year
    last_month_month = last_month_date.month

    last_month = [last_month_year, last_month_month]
    current_month = [current_year, current_month]

    return [last_month, current_month]

def cal_work_time(token):
    # 获取当前日期
    current_date = datetime.now()
    # 获取当前年份
    current_year = current_date.year
    # 获取当前周数
    current_week = int(current_date.strftime("%U"))
    # 获取当前周的起始日期
    start_date1, end_date1 = get_week_start_and_end_dates(current_year, current_week)
    print(start_date1, end_date1)
    # last_week_dict =

    start_date, end_date = get_current_week_range()
    start_date = start_date1

    date_range = get_dates_in_range(start_date, end_date)
    month_year = get_last_and_current_months()

    data = {i.strftime("%Y-%m-%d"): None for i in date_range}

    for year, month in month_year:
        month = str(month).zfill(2)
        year = str(year)
        request_(data, token, year, month)



    week_record = {"本周": {"工作天数": [], "工作时长": []}, "上周":{"工作天数": [], "工作时长": []}}
    # s = ""
    # print(end_date1)
    for date, times in data.items():
        if times is not None and '上班' in times and '下班' in times:
                # s += date + ", "
                start_time = times['上班']
                end_time = times['下班']
                seconds = calculate_work_duration(start_time, end_time)
                # print(date, end_date1)
                if date <= str(end_date1):


                    week_record["上周"]["工作时长"].append(seconds/3600)
                    week_record["上周"]["工作天数"].append(date)
                else:
                    week_record["本周"]["工作时长"].append(seconds / 3600)
                    week_record["本周"]["工作天数"].append(date)

    cur_week_time = f"""{np.mean(week_record["本周"]["工作时长"]):.3f}""" if week_record["本周"]["工作时长"] else "0"
    last_week_time = f"""{np.mean(week_record["上周"]["工作时长"]):.3f}""" if week_record["上周"]["工作时长"] else "0"

    return f"""
    本周工作天数: {','.join(week_record['本周']['工作天数'])}
    本周平均工作时长: {cur_week_time}小时
    上周工作天数: {','.join(week_record['上周']['工作天数'])}
    上周平均工作时长: {last_week_time}小时
    """


#     print(f"本周工作天数: {s[:-2]}")
#     print(f"平均工作时长: {np.mean(week_record):.3f} 小时")
#     return f"本周工作天数: {s[:-2]}", f"平均工作时长: {np.mean(week_record):.3f} 小时"
    
# if __name__ == "__main__":
#     print(cal_work_time(None))
# print("")