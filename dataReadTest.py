import mne
import matplotlib.pyplot as plt
import numpy as np

dataSource = './data/s01_060926_2n.set/s01_060926_2n.set'
event_id = {
    '251': 251,
    '252': 252,
    '253': 253,
    '254': 254
}


if __name__ == '__main__':
    raw = mne.io.read_raw_eeglab(dataSource)
    print(raw)
    # data, time = raw[:]
    # print('data:', data.shape)
    # print('time:', time.shape)

    # 读入的事件由annotations存储，依赖events_from_annotations将其转换为events类型
    # print(raw.annotations)
    # print(raw.annotations.onset[:6])

    # 将annotations转化为events
    events = mne.events_from_annotations(raw, event_id=event_id)
    print('ok')

    # 切出RT时间
    time_list = []
    for i in range(len(events[0])):
        # print(events[0][i], events[0][i][2])
        # if event[2] == 251 or event[2] == 252:
        start, end = 0, 0
        if events[0][i][2] == 254:
            continue
        elif events[0][i][2] == 253:
            continue
        elif events[0][i][2] == 251 or events[0][i][2] == 252:
            if events[0][i+1][2] == 253:
                start = events[0][i][0]
                end = events[0][i+1][0]
                time_list.append([start, end])
    # print(time_list)
    # print(len(time_list))

    # 计算RT时间
    time_spend = []
    for t in time_list:
        tmp = t[1] - t[0]
        if 100 < tmp < 15000:
            time_spend.append(tmp)
    alert_RT = np.percentile(time_spend, 5)
    # print(time_spend)
    # print(time_spend.__len__())

    # 标记疲劳情况
    # 0-fatigue, 1-alert, 2-other, 3-wrong
    fatigue_mark = []
    global_index = 90*500
    for i in range(len(time_list)):
        local_start = time_list[i][0] - global_index
        local_RT = time_list[i][1] - time_list[i][0]
        if 100 < local_RT < 15000:
            global_RT = 0
            global_count = 0
            for j in range(i):
                tmp = time_list[j][1] - time_list[j][0]
                if time_list[j][0] >= local_start and 100 < tmp < 15000:
                    global_RT += tmp
                    global_count += 1
            if global_count > 0:
                global_RT = global_RT / global_count
            else:
                global_RT = local_RT
            if local_RT <= 1.5 * alert_RT and global_RT <= 1.5 * alert_RT:
                fatigue_mark.append(1)
            elif local_RT >= 2.5 * alert_RT and global_RT >= 2.5 * alert_RT:
                fatigue_mark.append(0)
            else:
                fatigue_mark.append(2)
        else:
            fatigue_mark.append(3)

    # print(fatigue_mark)
    # print(len(fatigue_mark))

    # 统计标记数量
    count = [0, 0, 0, 0]
    for i in range(len(fatigue_mark)):
        count[fatigue_mark[i]] += 1
        if fatigue_mark[i] == 0 or fatigue_mark[i] == 1:
            print(time_list[i], time_list[i][1] - time_list[i][0])
    print(count)

    # 从时间取脑电信号
    # data, time = raw[:, time_list[0][0]:time_list[0][1]]
    # print(data)
    # print(time)
    # plt.plot(time, data[0])
    # plt.show()

    # 获取对应的疲劳和警觉的脑电信号
    sample_label = []
    for i in range(len(fatigue_mark)):
        # alert
        if fatigue_mark[i] == 1:
            data, _ = raw[:, time_list[i][0] - 3*500:time_list[i][0]]
            sample_label.append((data, 1))
        # fatigue
        elif fatigue_mark[i] == 0:
            data, _ = raw[:, time_list[i][0] - 3 * 500:time_list[i][0]]
            sample_label.append((data, 0))

    print(len(sample_label))

