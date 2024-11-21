from datetime import datetime, timedelta
import pandas as pd

def calculate_seconds_since_midnight(time_str):
    """计算给定时间自午夜以来的秒数"""
    time_parts = [int(part) for part in time_str.split(':')]
    return time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]

def calculate_total_seconds(start_date, start_time, end_date, end_time):
    """计算从开始日期时间到结束日期时间的总秒数"""
    start_datetime = datetime.strptime(f"{start_date} {start_time}", "%m/%d/%Y %H:%M:%S")
    end_datetime = datetime.strptime(f"{end_date} {end_time}", "%m/%d/%Y %H:%M:%S")
    
    # 总秒数差
    total_seconds_diff = (end_datetime - start_datetime).total_seconds()
    
    return total_seconds_diff

def calculate_row_numbers(start_date, start_time, end_date, end_time, start_base_date='10/9/2017 18:00:00'):
    """根据给定的开始和结束日期时间计算行号"""
    # 将开始基准时间转换为datetime
    base_datetime = datetime.strptime(start_base_date, "%m/%d/%Y %H:%M:%S")
    
    # 计算开始和结束时间自基准时间以来的总秒数
    start_seconds = calculate_total_seconds(start_base_date.split(' ')[0], start_base_date.split(' ')[1], start_date, start_time)
    end_seconds = start_seconds + calculate_total_seconds(start_date, start_time, end_date, end_time)
    
    # 计算行号，加1是因为行号从1开始
    start_row = int(start_seconds) + 1
    end_row = int(end_seconds) + 1
    
    return start_row, end_row


def mark_anomalies(output_csv_path, time_ranges):
    """读取CSV文件，根据多个时间段标记异常数据，并保存到新的CSV文件"""
    df = pd.read_csv('scripts/WADI_attackdata.csv')
    
    # 确保索引从1开始，以匹配行号
    df.index = range(1, len(df) + 1)
    
    df.loc[:, 'attack'] = 0 
    # 遍历每个时间范围并标记异常
    for start_date, start_time, end_date, end_time in time_ranges:
        start_row, end_row = calculate_row_numbers(start_date, start_time, end_date, end_time)
        # print(f"start_row: {start_row}, end_row: {end_row}")
        df.loc[start_row:end_row, 'attack'] = 1  # 更新最后一列'attack'，1代表异常
    
    # 保存更改后的文件
    df.to_csv(output_csv_path, index=False)
    print("异常数据标注完成并保存至新文件。")



# 输入时间范围示例
time_ranges = [
    ('10/9/2017', '19:25:00', '10/9/2017', '19:50:16'),
    ('10/10/2017', '10:24:10', '10/10/2017', '10:34:00'),
    ('10/10/2017', '10:55:00', '10/10/2017', '11:24:00'),
    ('10/10/2017', '11:07:46', '10/10/2017', '11:12:15'),
    ('10/10/2017', '11:30:40', '10/10/2017', '11:44:50'),
    ('10/10/2017', '13:39:30', '10/10/2017', '13:50:40'),
    ('10/10/2017', '14:48:17', '10/10/2017', '14:59:55'),
    ('10/10/2017', '14:53:44', '10/10/2017', '15:00:32'),
    ('10/10/2017', '17:40:00', '10/10/2017', '17:49:40'),
    ('10/11/2017', '10:55:00', '10/10/2017', '10:56:27'),
    ('10/11/2017', '11:17:54', '10/11/2017', '11:31:20'),
    ('10/11/2017', '11:36:31', '10/11/2017', '11:47:00'),
    ('10/11/2017', '11:59:00', '10/11/2017', '12:05:00'),
    ('10/11/2017', '12:07:30', '10/11/2017', '12:10:52'),
    ('10/11/2017', '12:16:00', '10/11/2017', '12:25:36'),
    ('10/11/2017', '15:26:30', '10/11/2017', '15:37:00')
]


mark_anomalies('scripts/WADI_attackdata_labelled.csv', time_ranges)
