import re

from matplotlib import pyplot as plt
from IO import Sample
import math
from sklearn.metrics import confusion_matrix
import itertools
import os
from tqdm import *
import Levenshtein
import time
import copy
import cv2
#
# io = PotIO()
# io.readFiles()

import numpy as np

'''
0:扼
1:遏
2:鄂
3:饿
4:恩
5:而
6:儿
7:耳
8:尔
9:饵
10:洱
11:二
12:贰
13:发
14:罚
15:筏
16:伐
17:乏
18:阀
19:法
20:藩
21:帆
22:番
23:翻
24:樊
25:矾
26:钒
27:繁
28:凡
29:烦
30:反
31:返
32:范
33:贩
34:犯
35:饭
36:泛
37:坊
38:芳
39:方
40:肪
41:房
42:防
43:妨
44:仿
45:访
46:纺
47:放
48:菲
49:非

['0xb6f3', '0xb6f4', '0xb6f5', '0xb6f6', '0xb6f7', '0xb6f8', '0xb6f9', '0xb6fa', '0xb6fb', '0xb6fc', '0xb6fd', '0xb6fe', '0xb7a1', '0xb7a2', '0xb7a3', '0xb7a4', '0xb7a5', '0xb7a6', '0xb7a7', '0xb7a8', '0xb7aa', '0xb7ab', '0xb7ac', '0xb7ad', '0xb7ae', '0xb7af', '0xb7b0', '0xb7b1', '0xb7b2', '0xb7b3', '0xb7b4', '0xb7b5', '0xb7b6', '0xb7b7', '0xb7b8', '0xb7b9', '0xb7ba', '0xb7bb', '0xb7bc', '0xb7bd', '0xb7be', '0xb7bf', '0xb7c0', '0xb7c1', '0xb7c2', '0xb7c3', '0xb7c4', '0xb7c5', '0xb7c6', '0xb7c7']
['扼', '遏', '鄂', '饿', '恩', '而', '儿', '耳', '尔', '饵', '洱', '二', '贰', '发', '罚', '筏', '伐', '乏', '阀', '法', '藩', '帆', '番', '翻', '樊', '矾', '钒', '繁', '凡', '烦', '反', '返', '范', '贩', '犯', '饭', '泛', '坊', '芳', '方', '肪', '房', '防', '妨', '仿', '访', '纺', '放', '菲', '非']

'''

data_fixed_length = 100  # this is the fixed length of vectors in a character
def augumentDataSet(dataset,data_fixed_length = 100,size=6):
    for i in range(len(dataset)):
        if len(dataset[i]) > data_fixed_length:
            dataset[i] = dataset[i][:data_fixed_length]
        else:
            dataset[i] = dataset[i] + [[0] * size] * (data_fixed_length - len(dataset[i]))
    return dataset

def toNpArr(s):
    new = []
    for i in s:
        temp = []
        for j in i:
            temp.append(np.asarray(j))
        new.append(np.asarray(temp))
    tmp = np.array(new)
    # tmp = tmp[:, np.newaxis]  # 给a最外层中括号中的每一个元素加[]
    a = tmp.shape
    # tmp = tmp.reshape(a[0],-1,6)
    return tmp


def defineDict(labels):
    label_dic = {}
    current_index = 0
    for l in labels:
        if l not in label_dic.keys():
            label_dic[l] = current_index
            current_index += 1
    return label_dic, {v: k for k, v in label_dic.items()}


def show(train_set, i):
    for stroke in train_set[i]:
        plot_circle([stroke[0], stroke[1]], 1, 'r')
        if stroke[-2] == 1:
            new_stroke = []
        if stroke[-1] == 1:
            new_stroke.append([stroke[0], stroke[1]])
            plt.plot([x[0] for x in new_stroke], [x[1] for x in new_stroke])
        else:
            new_stroke.append([stroke[0], stroke[1]])
    # pen_down_points, pen_change_points, pen_up_points = get_pen_down_change_up(train_set[i],0)
    # print("pen_down_points length is {}".format(len(pen_down_points)))
    # for pt in pen_down_points:
    #     tmp = [pt[0], pt[1]]
    #     plot_circle(tmp, 2, 'r')
    # for pt in pen_change_points:
    #     tmp = [pt[0], pt[1]]
    #     plot_circle(tmp, 3, 'b')
    # for pt in pen_up_points:
    #     tmp = [pt[0], pt[1]]
    #     plot_circle(tmp, 2, 'y')
    plt.show()
    plt.clf()

def save_show_strokes(train_set, i,save_path):
    plt.clf()
    for stroke in train_set[i]:
        if stroke[-2] == 1:
            new_stroke = []
        if stroke[-1] == 1:
            new_stroke.append([stroke[0], stroke[1]])
            plt.plot([x[0] for x in new_stroke], [x[1] for x in new_stroke])
        else:
            new_stroke.append([stroke[0], stroke[1]])
    pen_down_points, pen_change_points, pen_up_points = get_pen_down_change_up(train_set[i],0)
    print("pen_down_points length is {}".format(len(pen_down_points)))
    for pt in pen_down_points:
        tmp = [pt[0], pt[1]]
        plot_circle(tmp, 2, 'r')
    for pt in pen_change_points:
        tmp = [pt[0], pt[1]]
        plot_circle(tmp, 3, 'b')
    for pt in pen_up_points:
        tmp = [pt[0], pt[1]]
        plot_circle(tmp, 2, 'y')
    plt.savefig(save_path)

def getPts(train_set, i):
    pts = []
    for stroke in train_set[i]:
        pts.append((stroke[0], stroke[1]))
    return pts

def getUtf8(a):
    a = a.replace('0x', '')
    a_bytes = bytes.fromhex(a)
    # print(a_bytes)
    utf8_decode = a_bytes.decode("gbk")
    # print(utf8_decode)
    return utf8_decode

def allLined(test_set):
    for t in test_set:
        for l in t[1:-1]:
            l[4] = 0
            l[5] = 0

def test_result_by_hand_write(tag_code, tag, stroke_number, strokes_samples):
    '''
    :param tag_code:        标签gbk2312格式
    :param tag:             标签utf-8格式
    :param stroke_number:   笔画数，如：2
    :param strokes_sample:  笔画采样点列表，如：[[(x1,y1),(x2,y2)],[(x3,y3),(x4,y4)]]
    :return:
    '''
    # 将手写结果构造Sample对象
    sample = Sample(tag_code, tag, stroke_number, strokes_samples)
    sample.shrinkPixels()  # 以最小值对所有坐标进行平移，即减去最小值
    sample.normalize(128)  # 对笔画轨迹坐标进行规一化，参数128为坐标最大值，即右下角的坐标
    sample.removeRedundantPoints()  # 对靠得较近的笔画轨迹点进行去冗余
    # sample.show()
    # 将Sample对象转换成待测试样本
    # 组装[x,y,dx,dy,pen down, pen up]
    samples = [[0]]
    new_strokes = [[]]
    for stroke in strokes_samples:
        # [x,y,dx,dy,pen down, pen up]
        stroke_rep = []
        if len(stroke) == 1:
            stroke_rep = [[stroke[0][0], stroke[0][1], 0, 0, 1, 1]]
        else:
            for i in range(0, len(stroke) - 1):
                c = stroke[i]
                n = stroke[i + 1]
                stroke_rep.append([int(c[0]), int(c[1]), int(n[0] - c[0]), int(n[1] - c[1]), 0, 0])
            # pen down stroke
            stroke_rep[0] = [int(stroke_rep[0][0]), int(stroke_rep[0][1]), int(stroke_rep[0][2]),
                             int(stroke_rep[0][3]), 1, stroke_rep[0][5]]
            # pen up stroke
            stroke_rep[-1] = [int(stroke_rep[-1][0]), int(stroke_rep[-1][1]), int(stroke_rep[-1][2]),
                              int(stroke_rep[-1][3]), stroke_rep[-1][4], 1]
    #     new_strokes += stroke_rep
    # samples += [new_strokes[1:]]
    return stroke_rep

'''
计算笔划总长度，包括笔画之间的间隔长度
'''
def count_stroke_length(strokes):
    length = 0
    for i in range(1, len(strokes)):
        start = strokes[i - 1]
        end = strokes[i]
        dx = int(start[0]) - int(end[0])
        dy = int(start[1]) - int(end[1])
        # 用math.sqrt（）求平方根
        l = math.sqrt((dx ** 2) + (dy ** 2))
        length = length + l
    return length

'''
计算笔划总长度，不包括笔画之间的间隔长度
'''
def count_stroke_length_v2(strokes):
    length = 0
    for i in range(1, len(strokes)):
        start = strokes[i - 1]
        end = strokes[i]
        dx = int(start[0]) - int(end[0])
        dy = int(start[1]) - int(end[1])
        # 用math.sqrt（）求平方根
        if start[-1] != 1: # 如果不是两笔之间的间隔，即开始点不是收笔
            l = math.sqrt((dx ** 2) + (dy ** 2))
            length = length + l
    return length

'''
https://www.jb51.net/article/164697.htm
计算两条线段（向量）的夹角,v1:[x1,y1,x2,y2],即 点（x1,y1）到点（x1,y2）的线段
'''
def get_angle(v1, v2):
    dx1 = int(v1[2]) - int(v1[0])
    dy1 = int(v1[3]) - int(v1[1])
    dx2 = int(v2[2]) - int(v2[0])
    dy2 = int(v2[3]) - int(v2[1])
    # print(dx1,dx2,dy1,dy2)
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        # if included_angle > 180:
        #   included_angle = 360 - included_angle
    return included_angle

'''
获取所有的落笔、折笔、收笔点,type是配置是否要进行坐标转换，1转换，0不转换
'''
def get_pen_down_change_up(strokes,type=1):
    total_duration = 10
    stroke_length_total = count_stroke_length(strokes)  # 计算总长度
    pen_down_points = []
    pen_change_points = []
    pen_up_points = []
    for n, stroke in enumerate(strokes):
        if stroke[-2] == 1:  # 如果为落笔
            start = [stroke[0], stroke[1]]
            new_stroke = []
            new_stroke.append(start)
            pen_down_points.append(start)
        elif stroke[-1] == 1:  # 如果为收笔
            end = [stroke[0], stroke[1]]
            pen_up_points.append(end)

            # 判断要不要删除前一个折笔点
            if len(pen_change_points) > 0:
                stroke_tmp = strokes[n_change:n+1]
                # 用math.sqrt（）求平方根
                l = count_stroke_length(stroke_tmp)
                if l <=  14:
                    pen_change_points.remove(pen_change_points[-1])

        else:  # 中间笔划
            middle = [stroke[0], stroke[1]]
            gap = 10
            # v1 = [strokes[n - 1][0], strokes[n - 1][1], strokes[n][0], strokes[n][1]]
            # v2 = [strokes[n][0], strokes[n][1], strokes[n + 1][0], strokes[n + 1][1]]
            # 找middle点前面距离大于4的点
            tmp = new_stroke[::-1]
            for t in tmp:
                dx = int(t[0]) - int(middle[0])
                dy = int(t[1]) - int(middle[1])
                # 用math.sqrt（）求平方根
                l = math.sqrt((dx ** 2) + (dy ** 2))
                if l >= gap:
                    before = [t[0],t[1]]
                    break
            if l < gap:
               before = [new_stroke[-1][0],new_stroke[-1][1]]

            # 找middle点后面距离大于4的点
            tmp = strokes[n+1:]
            for t in tmp:
                dx = int(t[0]) - int(middle[0])
                dy = int(t[1]) - int(middle[1])
                # 用math.sqrt（）求平方根
                l = math.sqrt((dx ** 2) + (dy ** 2))
                if l >= gap or t[-1] == 1:
                    after = [t[0],t[1]]
                    break

            if l < gap:
                after = tmp[0]

            v1 = [before[0], before[1], strokes[n][0], strokes[n][1]]
            v2 = [strokes[n][0], strokes[n][1], after[0], after[1]]
            angle = get_angle(v1, v2)
            if angle > 50:  # 如果笔划折转度大于70
                new_stroke.append(middle)
                stroke_length = count_stroke_length(new_stroke)  # 计算当前段的笔划长度
                duration = total_duration * stroke_length / stroke_length_total  # 计算持续时长
                if duration > 0.3:
                    # print(v1)
                    # print(v2)
                    if type == 1:
                        v1[1], v1[3] = 120 - v1[1], 120 - v1[3]
                        v2[1], v2[3] = 120 - v2[1], 120 - v2[3]
                    xs = [v1[0], v1[2]]
                    ys = [v1[1], v1[3]]
                    if type == 1 or type == 0:
                        plt.plot(xs, ys,c='b',linewidth=6)

                    xs = [v2[0], v2[2]]
                    ys = [v2[1], v2[3]]
                    if type == 1 or type == 0:
                        plt.plot(xs, ys,c='b',linewidth=6)
                    pen_change_points.append(middle)
                    n_change = n
                    new_stroke = []
                    new_stroke.append(middle)
            else:
                new_stroke.append(middle)
    return pen_down_points,pen_change_points,pen_up_points

def plot_circle(center=(3, 3),r=0.5,color='k'):
  x = np.linspace(center[0] - r, center[0] + r, 5000)
  y1 = np.sqrt(r**2 - (x-center[0])**2) + center[1]
  y2 = -np.sqrt(r**2 - (x-center[0])**2) + center[1]
  plt.plot(x, y1, c=color)
  plt.plot(x, y2, c=color)

def plot_key_point(strokes):
    pen_down_points, pen_change_points, pen_up_points = get_pen_down_change_up(strokes)
    for pt in pen_down_points:
        tmp = [pt[0], 120 - pt[1]]
        plot_circle(tmp, 2, 'r')
    for pt in pen_change_points:
        tmp = [pt[0], 120 - pt[1]]
        plot_circle(tmp, 3, 'b')
    for pt in pen_up_points:
        tmp = [pt[0], 120 - pt[1]]
        plot_circle(tmp, 2, 'y')


def show_train_history(train_history, train_metrics, validation_metrics):
    plt.plot(train_history.history[train_metrics])
    plt.plot(train_history.history[validation_metrics])
    plt.title('Train History')
    plt.ylabel(train_metrics)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')


# 显示训练过程
def plot_history(history,user_defined_number,show_flag = True):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    show_train_history(history, 'acc', 'val_acc')
    plt.subplot(1, 2, 2)
    show_train_history(history, 'loss', 'val_loss')
    plt.savefig('./model_bak/' + str(user_defined_number) + "-val_acc.jpg")
    if show_flag:
        plt.show()

# https://blog.csdn.net/Nick_Dizzy/article/details/106412785
# https://www.jb51.net/article/188687.htm
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    predictions = model.predict_classes(x_val)
    truelabel = y_val.argmax(axis=-1)  # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel) + 1))


def labels_to_utf_8(labels):
    label_set = []
    for l in labels:
        if l not in label_set:
            label_set.append(l)
    result = []
    print(label_set)
    for a in label_set:
        a = a.replace('0x', '')
        a_bytes = bytes.fromhex(a)
        # print(a_bytes)
        utf8_decode = a_bytes.decode("gbk")
        result.append(utf8_decode)
    return result

'''
根据时间序列来组装样本数据
point_times：笔画落笔、折笔、收笔时间点
point_types:笔画类型，1为落笔，2为收笔，3为折笔
'''
def get_data_from_time_points(point_times,point_types,total_duration):
    tmp = [1 for p in point_types if p == 1]
    stroke_total = len(tmp)
    real_duration = point_times[-1] - point_times[0]
    rate = total_duration/real_duration
    unification_point_times = [rate*t for t in point_times]
    # unification_point_times.insert(0,0)
    stroke_rep = [[]]
    for i in range(len(point_times)):
        # 如果是落笔
        if point_types[i] == 1:
            pen_down, pen_change = 1, 0
            start_time = unification_point_times[i]
        # 如果是收笔
        elif point_types[i] == 2:
            duration = unification_point_times[i] - start_time  # 计算持续时长
            end_time = unification_point_times[i]
            pen_up = 1
            tmp = [[start_time, end_time, duration, pen_down, pen_change, pen_up, stroke_total]]
            stroke_rep += tmp
        # 如果是折笔
        elif point_types[i] == 3:
            duration = unification_point_times[i] - start_time  # 计算持续时长
            end_time = unification_point_times[i]
            pen_up = 0
            tmp = [[start_time, end_time, duration, pen_down, pen_change, pen_up, stroke_total]]
            stroke_rep += tmp
            pen_down, pen_change = 0, 1
            start_time = unification_point_times[i]
    result = stroke_rep[1:]
    mat_tmp = np.ndarray(shape=(1,), dtype=object)
    mat_tmp[0] = result
    return mat_tmp


'''
构造样本的特征值：（start_time,end_time,duration,pen_down,pen_change,pen_up,stroke_total）(超始时间点、结束时间点、距离下一采样点的时长、起点是否为落笔、终点是否为折笔点、起点是否为收笔、总笔画数)
sample_strokes 笔画采样点数据,
stroke_total 笔画总数,
total_duration  规范化的总时长
'''
def hand_features_by_one_sample(sample_strokes,total_duration,change_flag=True):
    # tmp = np.ndarray(shape=(1,), dtype=object)
    # tmp[0] = sample_strokes
    # show(tmp,0)
    stroke_total = len([1 for s in sample_strokes if s[-2] == 1 ])
    mat_tmp = np.ndarray(shape=(1,), dtype=object)
    strokes = sample_strokes  # 该样本的所有笔划数据
    p_down_points, p_change_points, p_up_points = get_pen_down_change_up(strokes,3) # 获取所有的落笔、折笔、收笔点
    # plot_key_point(strokes) # 显示所有的落笔、折笔、收笔点
    stroke_length_total = count_stroke_length(strokes) # 计算总长度
    # stroke_total = selected_stroke_total[i]   # 该样本的笔划总数
    if stroke_length_total == 0:
        print("ads")
    stroke_rep = [[]]
    for n, stroke in enumerate(strokes):
        stroke_length_sum = count_stroke_length(strokes[:n+1])  # 计算当前位置的累计笔划长度,即当前采样点到第一个点的积累距离
        start_time_on_each_point = total_duration * stroke_length_sum / stroke_length_total # 每个采样点的时间位置
        if stroke[-2] == 1:  # 如果为落笔
            start = [stroke[0], stroke[1]]
            pen_down, pen_change = 1, 0
            new_stroke = []
            new_stroke.append(start)
            start_time = start_time_on_each_point
        elif stroke[-1] == 1:  # 如果为收笔
            end_time = start_time_on_each_point
            duration = end_time - start_time
            pen_up = 1
            tmp = [[start_time, end_time, duration, pen_down, pen_change, pen_up,stroke_total]]
            stroke_rep += tmp
        else:  # 中间笔划
            if change_flag:
                middle = [stroke[0], stroke[1]]

                if middle in p_change_points: #如果是折笔点
                    end_time = start_time_on_each_point
                    duration = end_time - start_time
                    pen_up = 0
                    tmp = [[start_time, end_time, duration, pen_down, pen_change, pen_up, stroke_total]]
                    stroke_rep += tmp
                    pen_down, pen_change = 0, 1
                    new_stroke = []
                    new_stroke.append(middle)
                    start_time = start_time_on_each_point
    mat_tmp[0] = stroke_rep[1:]
    return mat_tmp

'''
构造样本的特征值：（start_time,end_time,duration,pen_down,pen_change,pen_up,direction）(超始时间点、结束时间点、距离下一采样点的时长、起点是否为落笔、终点是否为折笔点、起点是否为收笔、笔画走向)
sample_strokes 笔画采样点数据,
stroke_total 笔画总数,
total_duration  规范化的总时长
'''
def time_direction_features_by_one_sample(sample_strokes,total_duration,change_flag=True):
    # tmp = np.ndarray(shape=(1,), dtype=object)
    # tmp[0] = sample_strokes
    # show(tmp,0)
    stroke_total = len([1 for s in sample_strokes if s[-2] == 1 ])
    mat_tmp = np.ndarray(shape=(1,), dtype=object)
    strokes = sample_strokes  # 该样本的所有笔划数据
    p_down_points, p_change_points, p_up_points = get_pen_down_change_up(strokes,3) # 获取所有的落笔、折笔、收笔点
    # plot_key_point(strokes) # 显示所有的落笔、折笔、收笔点
    stroke_length_total = count_stroke_length(strokes) # 计算总长度
    # stroke_total = selected_stroke_total[i]   # 该样本的笔划总数
    if stroke_length_total == 0:
        print("ads")
        return None
    stroke_rep = [[]]
    for n, stroke in enumerate(strokes):
        stroke_length_sum = count_stroke_length(strokes[:n+1])  # 计算当前位置的累计笔划长度,即当前采样点到第一个点的积累距离
        start_time_on_each_point = total_duration * stroke_length_sum / stroke_length_total # 每个采样点的时间位置
        if stroke[-2] == 1:  # 如果为落笔
            start = [stroke[0], stroke[1]]
            pen_down, pen_change = 1, 0
            new_stroke = []
            new_stroke.append(start)
            start_time = start_time_on_each_point
        elif stroke[-1] == 1:  # 如果为收笔
            end_time = start_time_on_each_point
            duration = end_time - start_time
            pen_up = 1
            # dx = np.abs(new_stroke[-1][0] - stroke[0])
            # dy = np.abs(new_stroke[-1][1] - stroke[1])
            dx = int(new_stroke[-1][0]) - int(stroke[0])  # 起点到终点
            dy = int(new_stroke[-1][1]) - int(stroke[1])  # 起点到终点
            direction,direction1,direction2,direction3 = get_direction(dx,dy)
            # 如果前一笔是竖，当前笔是撇；或者一笔是竖，当前笔是捺，则合并这两笔，因为基本笔画中没这样的笔画
            if len(stroke_rep) > 1 and stroke_rep[-1][3] == 1 and int(stroke_rep[-1][-1]/10) == 2 and (int(direction/10) == 31 or int(direction/10) == 32):
                stroke_rep[-1][1] = end_time   # 更新结束时间
                stroke_rep[-1][2] = end_time - stroke_rep[-1][0]   # 更新持续时间
                stroke_rep[-1][-1] = direction # 更新笔画类型
            else:
                tmp = [[start_time, end_time, duration, pen_down, pen_change, pen_up,direction]]
                stroke_rep += tmp
        else:  # 中间笔划
            if change_flag:
                middle = [stroke[0], stroke[1]]

                if middle in p_change_points: #如果是折笔点
                    end_time = start_time_on_each_point
                    duration = end_time - start_time
                    pen_up = 0
                    # dx = np.abs(new_stroke[-1][0] - stroke[0])
                    # dy = np.abs(new_stroke[-1][1] - stroke[1])
                    dx = int(new_stroke[-1][0]) - int(stroke[0])  # 起点到终点
                    dy = int(new_stroke[-1][1]) - int(stroke[1])  # 起点到终点
                    direction,direction1,direction2,direction3 = get_direction(dx, dy)
                    tmp = [[start_time, end_time, duration, pen_down, pen_change, pen_up, direction]]
                    stroke_rep += tmp
                    pen_down, pen_change = 0, 1
                    new_stroke = []
                    new_stroke.append(middle)
                    start_time = start_time_on_each_point
    mat_tmp[0] = stroke_rep[1:]
    return mat_tmp

'''
构造样本的特征值：（start_time,end_time,duration,pen_down,pen_up,stroke_type,stroke_total）(超始时间点、结束时间点、距离下一采样点的时长、起点是否为落笔、起点是否为收笔、笔画种类，笔画总数)
sample_strokes 笔画采样点数据,
stroke_total 笔画总数,
total_duration  规范化的总时长
'''
def time_stroke_order_features_by_one_sample(sample_strokes,total_duration,stroke_order):
    # tmp = np.ndarray(shape=(1,), dtype=object)
    # tmp[0] = sample_strokes
    # show(tmp,0)
    stroke_total = len([1 for s in sample_strokes if s[-1] == 1 ])
    mat_tmp = np.ndarray(shape=(1,), dtype=object)
    strokes = sample_strokes  # 该样本的所有笔划数据
    # p_down_points, p_change_points, p_up_points = get_pen_down_change_up(strokes,3) # 获取所有的落笔、折笔、收笔点
    # plot_key_point(strokes) # 显示所有的落笔、折笔、收笔点
    # stroke_length_total = count_stroke_length(strokes) # 计算总长度
    # # stroke_total = selected_stroke_total[i]   # 该样本的笔划总数
    # if stroke_length_total == 0 or stroke_total != len(stroke_order):
    #     print("ads,{},{},{}".format(stroke_length_total,stroke_total,len(stroke_order)))
    #     return None
    # stroke_rep = [[]]
    # stroke_index = 0
    # for n, stroke in enumerate(strokes):
    #     stroke_length_sum = count_stroke_length(strokes[:n+1])  # 计算当前位置的累计笔划长度,即当前采样点到第一个点的积累距离
    #     start_time_on_each_point = total_duration * stroke_length_sum / stroke_length_total # 每个采样点的时间位置
    #     if stroke[-2] == 1:  # 如果为落笔
    #         start_time = start_time_on_each_point
    #     elif stroke[-1] == 1:  # 如果为收笔
    #         end_time = start_time_on_each_point
    #         duration = end_time - start_time
    #         pen_down,pen_up = 1,1
    #         stroke_type = stroke_order[stroke_index]
    #         stroke_index += 1
    #         tmp = [[0, 0, duration, pen_down, pen_up,stroke_type,stroke_total]] # 只考虑每笔的持续时长
    #         stroke_rep += tmp

    # 另一种特征形式，只算每个笔画的持续时长
    stroke_length_total = count_stroke_length_v2(strokes)  # 计算总长度，不包括笔画之间的间隔
    # stroke_total = selected_stroke_total[i]   # 该样本的笔划总数
    if stroke_length_total == 0 or stroke_total != len(stroke_order):
        print("ads,{},{},{}".format(stroke_length_total, stroke_total, len(stroke_order)))
        return None
    stroke_rep = [[]]
    stroke_index = 0
    for n, stroke in enumerate(strokes):
        if stroke[-2] == 1:  # 如果为落笔
            start_index = n
        elif stroke[-1] == 1:  # 如果为收笔
            current_stroke_length = count_stroke_length(strokes[start_index:n + 1])  # 计算当前位置的累计笔划长度,即当前采样点到第一个点的积累距离
            duration = total_duration * current_stroke_length / stroke_length_total
            pen_down, pen_up = 1, 1
            stroke_type = stroke_order[stroke_index]
            stroke_index += 1
            # tmp = [[start_time, end_time, duration, pen_down, pen_up,stroke_type,stroke_total]]
            tmp = [[0, 0, duration, pen_down, pen_up, stroke_type, stroke_total]]  # 只考虑每笔的持续时长
            stroke_rep += tmp
    mat_tmp[0] = stroke_rep[1:]
    return mat_tmp

def get_direction(dx,dy):
    dx_abs = np.abs(dx)
    dy_abs = np.abs(dy)
    if dy_abs == 0:
        direction1 = 1
    elif dx_abs/dy_abs >= 3.5:  # 横向
        direction1 = 1
    elif dx_abs/dy_abs <= 0.3:  # 竖向
        direction1 = 2
    else:               #即斜向
        direction1 = 3
    direction2 = 1 if dx > 0 else 2  # 向左为1，向右为2
    direction3 = 1 if dy > 0 else 2  # 向下为1，向上为2
    # 横笔只管是向左还是向右，竖笔只管是向上还是向下
    if direction1 == 1:
        direction = direction1 * 10 + direction2
    elif direction1 == 2:
        direction = direction1 * 10 + direction3
    else:
        direction = direction1 * 100 + direction2 * 10 + direction3
    return direction,direction1,direction2,direction3

def gbk2312_to_utf8(gbk2312):
    a = gbk2312.replace('0x', '')
    a_bytes = bytes.fromhex(a)
    # print(a_bytes)
    utf8_decode = a_bytes.decode("gbk")
    return utf8_decode

def utf8_to_gbk2312(utf8):
    b = utf8.encode("gb2312")
    bb = b.hex()
    # print(a_bytes)
    gbk2312_decode = '0x' + bb
    return gbk2312_decode

def draw_hand_features(hand_features):
    # hand_feature = hand_features[0]
    ymin = 1
    for i, hf in enumerate(hand_features):
        if hf[3] == 1 or hf[4] == 1:
            ys = [1,1]
            xs = [hf[0],hf[1]]

            plt.plot(xs,ys)
            ymin += 1
    plt.show()

def save_show_hand_features(hand_features,save_path):
    # hand_feature = hand_features[0]
    plt.clf()
    ymin = 1
    for i, hf in enumerate(hand_features):
        if hf[3] == 1 or hf[4] == 1:
            ys = [1,1]
            xs = [hf[0],hf[1]]

            plt.plot(xs,ys)
            ymin += 1
    plt.savefig(save_path)

def get_time_points_from_hand_features(hand_features):
    point_times = []
    point_types = []
    for hf in hand_features:
        if hf[0] not in point_times:
            point_times.append(hf[0])
        if hf[1] not in point_times:
            point_times.append(hf[1])
        if hf[3] == 1:
            point_types.append(1)
        if hf[4] == 1:
            point_types.append(3)
        if hf[5] == 1:
            point_types.append(2)
    return point_times,point_types

def cv2_img_add_text(img, text, left, top, text_color=(0, 255, 0), text_size=20):
    """
    :param img:
    :param text:
    :param left:
    :param top:
    :param text_color:
    :param text_size
    :return:
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_text = ImageFont.truetype(
        "./assets/simsun.ttc", text_size, encoding="utf-8")  # 使用宋体
    draw.text((left, top), text, text_color, font=font_text)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def find_MostNewFile(path):
    # 获取文件夹中的所有文件
    lists = os.listdir(path)
    # 对获取的文件根据修改时间进行排序
    lists.sort(key=lambda x: os.path.getmtime(path + '\\' + x))
    # 把目录和文件名合成一个路径
    file_new = os.path.join(path, lists[-1])
    return file_new

'''
筛选gbk2312一级和二级汉字
https://www.qqxiuzi.cn/zh/hanzi-gb2312-bianma.php
GB2312编码是第一个汉字编码国家标准，由中国国家标准总局1980年发布，1981年5月1日开始使用。GB2312编码共收录汉字6763个，其中一级汉字3755个，二级汉字3008个。同时，GB2312编码收录了包括拉丁字母、希腊字母、日文平假名及片假名字母、俄语西里尔字母在内的682个全角字符。

分区表示
GB2312编码对所收录字符进行了“分区”处理，共94个区，每区含有94个位，共8836个码位。这种表示方式也称为区位码。
01-09区收录除汉字外的682个字符。
10-15区为空白区，没有使用。
16-55区收录3755个一级汉字，按拼音排序。
56-87区收录3008个二级汉字，按部首/笔画排序。
88-94区为空白区，没有使用。
举例来说，“啊”字是GB2312编码中的第一个汉字，它位于16区的01位，所以它的区位码就是1601。

双字节编码
GB2312规定对收录的每个字符采用两个字节表示，第一个字节为“高字节”，对应94个区；第二个字节为“低字节”，对应94个位。所以它的区位码范围是：0101－9494。区号和位号分别加上0xA0就是GB2312编码。例如最后一个码位是9494，区号和位号分别转换成十六进制是5E5E，0x5E+0xA0＝0xFE，所以该码位的GB2312编码是FEFE。

GB2312编码范围：A1A1－FEFE，其中汉字的编码范围为B0A1-F7FE，第一字节0xB0-0xF7（对应区号：16－87），第二个字节0xA1-0xFE（对应位号：01－94）。
'''
def get_gbk2312_1_2(data_set,labels,stroke_total):
    selected_data_set = []
    selected_labels = []
    stroke_number = []
    c10 = 'B0A1' # 一级汉字起始位置
    int10 = 45217
    c11 = 'D7F0'  # 一级汉字结束位置
    int11 = 55289

    c20 = 'D8A1'  # 二级汉字起始位置
    int20 = 55457
    c21 = 'F7FE'
    int21 = 63486 # 二级汉字结束位置
    for i in range(len(data_set)):
        chars_gbk2312 = labels[i]
        chars_gbk2312 = chars_gbk2312.replace("0x",'')
        a = int(chars_gbk2312, 16)
        if int10 <=  a <= int21:  # 如果标准笔划数 == 实际书写笔划数，则记录下来
            # print("{},{},{},{}".format(i, utf8_decode, s_total,stroke_number[i]))
            if len(data_set[i]) > 1:  # 去掉个别脏数据
                selected_data_set.append(data_set[i])
                selected_labels.append(labels[i])
                stroke_number.append(stroke_total[i])
    return selected_data_set, selected_labels,stroke_number


def data_augument_by_direction(data_set,labels,type=1):
    import copy
    import random
    lamda = 6  # 增强系数
    mat_tmp = np.ndarray(shape=(lamda*len(data_set),), dtype=object)
    label_mat = np.ndarray(shape=(lamda * len(data_set),), dtype=object)
    data_set_result = []
    labels_result = []
    index = 0
    for i in tqdm(range(len(data_set))):
        d = data_set[i]
        mat_tmp[index] = d
        label_mat[index] = labels[i]
        # data_set_result.append(d)
        # labels_result.append(labels[i])
        index += 1
        for j in range(1,lamda):
            d_tmp = copy.deepcopy(d)
            for n,t in enumerate(d):
                if type == 1:
                    if t[-1]/100 == 3: # 如果是斜向，则随机改变为横向或竖向或保持不变
                        r = random.random()
                        # d_tmp[n][-1] = 1 if r <= 0.33 else 2 if r<= 0.66 else 3
                        # d_tmp[n][-1] = 1 if r <= 0.33 else 2
                        d_tmp[n][-1] = t[-1] - 200 if r <= 0.5 else t[-1] - 100
                else:
                    if t[-2] == 3: # 如果笔画类型（即倒序第二t[-2]）是撇，则改变为竖
                        r = random.random()
                        # d_tmp[n][-1] = 1 if r <= 0.33 else 2 if r<= 0.66 else 3
                        # d_tmp[n][-1] = 1 if r <= 0.33 else 2
                        d_tmp[n][-2] = 2 if r <= 0.5 else t[-2]
                    if t[-2] == 4: # 如果笔画类型（即倒序第二t[-2]）是捺，则随机改变为横向或竖向或保持不变
                        r = random.random()
                        # d_tmp[n][-1] = 1 if r <= 0.33 else 2 if r<= 0.66 else 3
                        # d_tmp[n][-1] = 1 if r <= 0.33 else 2
                        d_tmp[n][-2] = 1 if r <= 0.33 else 2 if r<= 0.66 else t[-2]
            mat_tmp[index] = d_tmp
            label_mat[index] = labels[i]
            # data_set_result.append(d)
            # labels_result.append(labels[i])
            index += 1
    return mat_tmp,label_mat
    # return data_set_result,labels_result
'''
随机修改两笔的顺序
特征值：（start_time,end_time,duration,pen_down,pen_up,stroke_type,stroke_total）(超始时间点、结束时间点、距离下一采样点的时长、起点是否为落笔、起点是否为收笔、笔画种类，笔画总数)
'''
def data_augument_by_change_order(data_set,labels):
    import copy
    import random
    lamda = 3  # 增强系数
    mat_tmp = np.ndarray(shape=(lamda*len(data_set),), dtype=object)
    label_mat = np.ndarray(shape=(lamda * len(data_set),), dtype=object)
    index = 0
    for i in tqdm(range(len(data_set))):
        d = data_set[i]
        mat_tmp[index] = d
        label_mat[index] = labels[i]
        # data_set_result.append(d)
        # labels_result.append(labels[i])
        index += 1
        for j in range(1,lamda):
            d_tmp = copy.deepcopy(d)
            if len(d) > 1:
                if len(d) == 2: #修改两笔的顺序
                    r = 0
                elif len(d) > 2: # 随机修改两笔的顺序
                    r = random.randint(0, len(d) - 2)
                start = d_tmp[r][0]  # 开始点
                duration1 = d_tmp[r][2]  # 第一笔的时长
                duration2 = d_tmp[r+1][2]  # 第二笔的时长
                gap = d_tmp[r+1][0] - d_tmp[r][1]  # 两份笔的间隔时长
                type1 = d_tmp[r][-2]
                type2 = d_tmp[r+1][-2]
                d_tmp[r][0],d_tmp[r][1], d_tmp[r][2], d_tmp[r][-2] = start, start + duration2, duration2, type2
                d_tmp[r+1][0], d_tmp[r+1][1], d_tmp[r+1][2], d_tmp[r+1][-2] = start + duration2 + gap, start + duration2 + gap + duration1, duration1, type1

            mat_tmp[index] = d_tmp
            label_mat[index] = labels[i]
            # data_set_result.append(d)
            # labels_result.append(labels[i])
            index += 1
    return mat_tmp,label_mat
    # return data_set_result,labels_result

'''
删除最后的二或三笔
特征值：（start_time,end_time,duration,pen_down,pen_up,stroke_type,stroke_total）(超始时间点、结束时间点、距离下一采样点的时长、起点是否为落笔、起点是否为收笔、笔画种类，笔画总数)
'''
def data_augument_by_remove_stroke(data_set,labels):
    import copy
    import random
    lamda = 2  # 增强系数
    mat_tmp = np.ndarray(shape=(lamda*len(data_set),), dtype=object)
    label_mat = np.ndarray(shape=(lamda * len(data_set),), dtype=object)
    index = 0
    for i in tqdm(range(len(data_set))):
        d = data_set[i]
        mat_tmp[index] = d
        label_mat[index] = labels[i]
        # data_set_result.append(d)
        # labels_result.append(labels[i])
        index += 1
        for j in range(1,lamda):
            d_tmp = copy.deepcopy(d)
            # if 6 < len(d) <= 10: # 删除最后一笔
            #     d_tmp = d_tmp[:-1]
            #     n = -1
            if 20 < len(d) <= 25:  # 删除最后一或二笔
                r = random.random()
                n = -2 if r <= 0.5 else -1
                d_tmp = d_tmp[:n]
            if 25 < len(d):  # 删除最后二或三笔
                r = random.random()
                n = -3 if r <= 0.5 else -2
                d_tmp = d_tmp[:n]

            if len(d) > 20:
                for d in d_tmp:
                    d[-1] = d[-1] + n

            mat_tmp[index] = d_tmp
            label_mat[index] = labels[i]
            # data_set_result.append(d)
            # labels_result.append(labels[i])
            index += 1
    return mat_tmp,label_mat

'''
根据时间序列来组装样本的特征值：（start_time,end_time,duration,pen_down,pen_change,pen_up,direction）(超始时间点、结束时间点、距离下一采样点的时长、起点是否为落笔、终点是否为折笔点、起点是否为收笔、笔画走向)
point_times：笔画落笔、折笔、收笔时间点
point_types:笔画类型，1为横笔，2为竖笔，3为斜笔，0为收笔
'''
def get_time_direction_features_from_time_points(point_times,point_types,total_duration):
    real_duration = point_times[-1] - point_times[1]
    rate = total_duration/real_duration
    unification_point_times = [rate*t for t in point_times]
    # unification_point_times.insert(0,0)
    stroke_rep = [[]]
    for i in range(1,len(point_times)):
        # 如果前一笔是收笔，则当前笔是落笔
        if point_types[i-1] == 0:
            pen_down, pen_change = 1, 0
            start_time = unification_point_times[i]
            direction = point_types[i]
        # 如果是收笔
        elif point_types[i] == 0:
            duration = unification_point_times[i] - start_time  # 计算持续时长
            end_time = unification_point_times[i]
            pen_up = 1
            tmp = [[start_time, end_time, duration, pen_down, pen_change, pen_up, direction]]
            stroke_rep += tmp
        # 如果前一笔不是收笔，且与当前不折笔
        else:
            duration = unification_point_times[i] - start_time  # 计算持续时长
            end_time = unification_point_times[i]
            pen_up = 0
            tmp = [[start_time, end_time, duration, pen_down, pen_change, pen_up, direction]]
            stroke_rep += tmp
            pen_down, pen_change = 0, 1
            start_time = unification_point_times[i]
    result = stroke_rep[1:]
    mat_tmp = np.ndarray(shape=(1,), dtype=object)
    mat_tmp[0] = result
    return mat_tmp


def check_dtw_in_topN(one_test,standarded_time_series,max_N_idx):
    from dtw_test import get_time_series_from_one_sample
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    time_series = get_time_series_from_one_sample(one_test)
    new_max_N_idx = [key for key in max_N_idx if key in standarded_time_series]
    if len(new_max_N_idx) == 0:
        return max_N_idx[0]
    min_key = -1
    min_distance = 10000
    for key in new_max_N_idx:
        other = standarded_time_series[key]
        # distance, paths, max_sub = TimeSeriesSimilarityImprove(time_series, other)
        distance, path = fastdtw(time_series, other, dist=euclidean)
        if distance < min_distance:
            min_key = key
            min_distance = distance
    return min_key

def check_stroke_order_in_topN(stroke_order,k2l,max_N_idx,all_stroke_orders):
    labels = [k2l[key] for key in max_N_idx]
    char_max_N_idx = [getUtf8(key) for key in labels]
    print(char_max_N_idx)
    stroke_order_max_N_idx = [all_stroke_orders[key] for key in char_max_N_idx]
    # min_lcseque = find_lcseque(stroke_order,stroke_order_max_N_idx[0])
    min_lcseque,m = find_lcsubstr(stroke_order,stroke_order_max_N_idx[0])
    best_key = max_N_idx[0]
    for i in range(1,len(stroke_order_max_N_idx)):
        so = stroke_order_max_N_idx[i]
        # lcseque = find_lcseque(so,stroke_order)
        lcseque,m = find_lcsubstr(so,stroke_order)
        if len(min_lcseque) < len(lcseque):
            best_key = max_N_idx[i]
            min_lcseque = lcseque

    return best_key,char_max_N_idx


'''
1为横
2为竖
3为撇
4为捺
5为折

样本的特征值：（start_time,end_time,duration,pen_down,pen_change,pen_up,direction）(超始时间点、结束时间点、距离下一采样点的时长、起点是否为落笔、终点是否为折笔点、起点是否为收笔、笔画走向)
'''
def get_stroke_order_from_time_direction_features(time_direction_features):
    stroke_order = []
    for tdf in time_direction_features:
        if tdf[-2] == 1: # 如果为收笔
            if tdf[3] == 1: # 如果为落笔
                stroke_order.append(tdf[-1])  # 这个笔不为折笔，则直接获取笔划类型
            else:
                stroke_order.append(5)  # 这个笔为折笔
    result = "".join('%s' %id for id in stroke_order)
    return result

def get_all_stroke_orders(strokes_path):
    all_stroke_orders = {}
    with open(strokes_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            tmp = line.split(":")
            char = tmp[0].replace('"','')
            stroke_order = tmp[1].split(',')[0].replace('"','')
            all_stroke_orders[char] = stroke_order
    return all_stroke_orders


def find_lcseque(s1, s2):
    # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    # d用来记录转移方向
    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 'ok'
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 'left'
            else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    s = []
    while m[p1][p2]:  # 不为None时
        c = d[p1][p2]
        if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == 'left':  # 根据标记，向左找下一个
            p2 -= 1
        if c == 'up':  # 根据标记，向上找下一个
            p1 -= 1
    s.reverse()
    return ''.join(s)

def find_lcsubstr(s1, s2):
    m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]  #生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax=0   #最长匹配的长度
    p=0  #最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                m[i+1][j+1]=m[i][j]+1
                if m[i+1][j+1]>mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return s1[p-mmax:p],mmax   #返回最长子串及其长度

def get_stroke_total(strokes_path):
    strokes = {}
    with open(strokes_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            tmp = line.split(",")
            char = tmp[1]
            total = tmp[2].split("\t")[0]
            strokes[char] = int(total)
    return strokes

def check_xy_status(x,y,ts,dxs,dys,pen_status,pen_times,char=None,stroke_num= None,sample_save_path=None,print_flag=False):
    import scipy.signal as signal
    import time
    PEN_UP = 0
    PEN_DOWN = 100
    XY_MAX_LEN = 20
    xy_theshold = 200

    dx = x - 500
    dxs = [dx] + dxs[:-1]
    dy = y - 500
    dys = [dy] + dys[:-1]

    window_width = 5

    t = round(time.time(), 3)
    ts = [t] + ts[:-1]
    if len(dxs) >= XY_MAX_LEN:
        if (np.min(np.abs(dxs[:window_width])) > xy_theshold or np.min(np.abs(dys[:window_width])) > xy_theshold ) and (np.max(np.abs(dxs[window_width + 1:2 * XY_MAX_LEN])) < xy_theshold and np.max(np.abs(dys[XY_MAX_LEN + 1 : 2 * XY_MAX_LEN])) < xy_theshold):
            if len(pen_status) == 0 or pen_status[0] != PEN_DOWN:
                print("begin=======================================")
                pen_status.appendleft(PEN_DOWN)
                pen_times.appendleft(t)



        if (np.max(np.abs(dxs[:7*window_width]))) < xy_theshold and (np.max(np.abs(dys[:7*window_width]))) < xy_theshold:
            if len(pen_status) > 0 and pen_status[0] != PEN_UP:
                print("end===============")
                pen_status.appendleft(PEN_UP)
                pen_times.appendleft(t)
                start_time = pen_times[-1]
                end_time = t
                if start_time in ts and end_time in ts:
                    start_index = ts.index(start_time)+ window_width + 1
                    end_index = ts.index(end_time)
                    # print(start_index)
                    # print(end_index)
                    # print(dxs[start_index:end_index:-1])

                    tt = ts[end_index:start_index]
                    tt = tt[::-1]
                    x_tmp = dxs[end_index:start_index]
                    x_tmp = x_tmp[::-1]
                    x_tmp = signal.medfilt(x_tmp, 3)
                    # print(x_tmp)
                    y_tmp = dys[end_index:start_index]
                    y_tmp = y_tmp[::-1]
                    y_tmp = signal.medfilt(y_tmp, 3)
                    # print(x_tmp)
                    # print(y_tmp)
                    # print(tt)
                    pen_types, point_times, time_types, indexs = slice_times_serials(x_tmp,y_tmp,tt)
                    stroke_order = ["横" if int(p/100) == 1 else "竖" if int(p/100) == 2 else "撇" if int(p/10) == 31 else "捺" if int(p/10) == 32 else "折" if p == 5 else "N" for p in pen_types]
                    # print(stroke_order)
                    # print(pen_types)
                    # print(point_times)
                    # print(time_types)
                    start_indexs = [indexs[i] for i in range(len(indexs)) if time_types[i] == 1]
                    end_indexs = [indexs[i] for i in range(len(indexs)) if time_types[i] == 2]
                    change_indexs = [indexs[i] for i in range(len(indexs)) if time_types[i] == 3]
                    # print(indexs)
                    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签 https://blog.csdn.net/weixin_41767802/article/details/108047350
                    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
                    plt.title(char,fontsize='xx-large',fontweight='heavy')
                    plt.xlabel("stroke_num is {},start_num is {}".format(stroke_num,len(start_indexs)))
                    plt.ylabel(char)
                    plt.plot(x_tmp,'-r*', label = 'xs %')
                    plt.plot(y_tmp,'-g*', label = 'ys %')
                    plt.vlines(start_indexs,0,400,'r',':')
                    plt.vlines(end_indexs, 0, 400,'black',':')
                    plt.vlines(change_indexs, 0, 300,'g',':' )
                    plt.legend(loc='upper right', fontsize=10)  # 标签位置
                    plt.savefig("./tmp/" + str(np.round(time.time(),2)) + ".png")
                    plt.pause(0.5)
                    plt.close()
                    pen_times.clear()
                    pen_status.clear()
                    print_flag = True
                    if sample_save_path is not None:
                        with open(sample_save_path, mode='a') as filename:
                            filename.write(char)
                            filename.write('\n')  # 换行
                            x_tmp = ' '.join(str(i).replace("\n",'') for i in x_tmp)
                            filename.write(x_tmp)
                            filename.write('\n')  # 换行
                            y_tmp = ' '.join(str(i).replace("\n", '') for i in y_tmp)
                            filename.write(y_tmp)
                            filename.write('\n')  # 换行
                            start_indexs = ' '.join(str(i).replace("\n", '') for i in start_indexs)
                            filename.write(start_indexs)
                            filename.write('\n')  # 换行
                            end_indexs = ' '.join(str(i).replace("\n", '') for i in end_indexs)
                            filename.write(end_indexs)
                            filename.write('\n')  # 换行
                            # change_indexs = ' '.join(str(i).replace("\n", '') for i in change_indexs)
                            # filename.write(change_indexs)
                            # filename.write('\n')  # 换行

    return ts,dxs, dys,pen_status,pen_times,print_flag

def slice_times_serials(xs,ys,tt):
    xs = [x if np.abs(x) > 20 else 0 for x in xs]   # 阀值化
    ys = [y if np.abs(y) > 20 else 0 for y in ys]   # 阀值化
    value_theshold = 100
    time_theshold = 0.01
    time_theshold_changeed = 0.18
    rate_theshold = 0.15
    slice_pen_times = [tt[0]]
    slice_pen_types = []      # 1为横笔，2为竖笔，3为斜笔
    time_types = [1]  # 1为落笔点，2为收笔点，3，为折笔
    indexs = [0]

    gap_xy = [[np.abs(xs[i] - ys[i]),i] for i in range(len(xs)) if xs[i] > 200 and ys[i] > 200]
    min_gap_xy_index = [gap_xy[i][1] for i in range(1,len(gap_xy)-1) if gap_xy[i-1][0] > gap_xy[i][0] < gap_xy[i+1][0]]  # x、y包络线的交会点
    # print(xs)
    # print(ys)

    for i in range(2,len(xs)-5):
        x = xs[i]
        y = ys[i]

        # if np.abs(x - value_theshold) < 100 or np.abs(y - value_theshold) < 100: # 落笔和收笔时间点
        # if (np.abs(x) < 10 and np.abs(xs[i+1]) > 20) or (np.abs(y) < 10 and np.abs(ys[i+1]) >20):  # 落笔点
        #     if tt[i] - slice_pen_times[-1] > time_theshold:  #间隔大于阀值，防止重复计入
        #         slice_pen_times.append(tt[i])
        #         indexs.append(i)
        #         if time_types[-1] == 1: #如果前一条线也落笔点，则需要将本笔修改为折笔
        #             time_types.append(3)
        #         else:
        #             time_types.append(1)
        # if (np.abs(x) > 20 and np.abs(xs[i+1]) < 10) or (np.abs(y) > 20 and np.abs(ys[i+1]) < 10):  # 收笔点
        #     print(tt[i] - slice_pen_times[-1])
        #     if tt[i] - slice_pen_times[-1] > time_theshold:  #间隔大于阀值，防止重复计入
        #         slice_pen_times.append(tt[i+1])
        #         indexs.append(i+1)
        #         if time_types[-1] == 2: #如果前一条线也收笔点，则需要修改为折笔
        #             time_types[-1] = 3
        #         time_types.append(2)
        #
        # if (np.abs(x) < 10 and np.abs(xs[i+1]) > 20) or (np.abs(y) < 10 and np.abs(ys[i+1]) >20):  # 落笔点
        #     if tt[i] - slice_pen_times[-1] > time_theshold:  #间隔大于阀值，防止重复计入
        #         slice_pen_times.append(tt[i])
        #         indexs.append(i)
        #         if time_types[-1] == 1: #如果前一条线也落笔点，则需要将本笔修改为折笔
        #             time_types.append(3)
        #         else:
        #             time_types.append(1)
        # if (np.abs(x) > 300 and np.abs(xs[i+2]) < 10 and np.abs(xs[i+3]) < 10) or (np.abs(y) > 300 and np.abs(ys[i+2]) < 10 and np.abs(ys[i+3]) < 10):  # 收笔点，当前大于200，随后两个小于10
        #     # print(tt[i] - slice_pen_times[-1])
        #     slice_pen_times.append(tt[i+1])
        #     indexs.append(i+1)
        #     if time_types[-1] == 2: #如果前一条线也收笔点，则需要修改为折笔
        #         time_types[-1] = 3
        #     time_types.append(2)
        #
        #
        # if (np.abs(xs[i]) > 0 and np.max(np.abs(xs[i+1:])) == 0 and np.max(np.abs(ys[i+1:])) == 0) or (np.abs(ys[i]) > 0 and np.max(np.abs(ys[i+1:])) == 0 and np.max(np.abs(xs[i+1:])) == 0 ):  # 最后一个收笔点
        #     slice_pen_times.append(tt[i+1])
        #     indexs.append(i+1)
        #     if time_types[-1] == 2: #如果前一条线也收笔点，则需要修改为折笔
        #         time_types[-1] = 3
        #     time_types.append(2)

        # 判断起点，判断规则是：当前点及前一点为静止，后一点为运动，另一包络线的当前点及前一点为静止
        # if (np.max(np.abs(xs[i-1:i+1])) < 30 and np.abs(xs[i+1]) > 30 and np.max(np.abs(ys[i-1:i+1])) < 30) or (np.max(np.abs(ys[i-1:i+1])) < 30 and np.abs(ys[i+1]) > 30 and np.max(np.abs(xs[i-1:i+1])) < 30) : # 落笔点
        # 判断起点，判断规则是：当前点为静止，后一点为运动，另一包络线的当前点及前一点为静止
        if (np.max(np.abs(xs[i])) < 30 and np.abs(xs[i+1]) > 30 and np.max(np.abs(ys[i-1:i+1])) < 30) or (np.max(np.abs(ys[i])) < 30 and np.abs(ys[i+1]) > 30 and np.max(np.abs(xs[i-1:i+1])) < 30) : # 落笔点
            if tt[i] - slice_pen_times[-1] > time_theshold:  #间隔大于阀值，防止重复计入
                slice_pen_times.append(tt[i])
                indexs.append(i)
                if time_types[-1] == 1: #如果前一条线也落笔点，则需要将本笔修改为折笔
                    time_types.append(3)
                else:
                    time_types.append(1)
        # 判断终点，判断规则是：当前点及后一点为静止，前一点为运动，另一包络线的当前点及后一点为静止
        if (np.max(np.abs(xs[i:i+2])) < 30 and np.abs(xs[i-1]) > 30 and np.max(np.abs(ys[i:i+2])) < 30) or (np.max(np.abs(ys[i:i+2])) < 30 and np.abs(ys[i-1]) > 30 and np.max(np.abs(xs[i:i+2])) < 30):  # 收笔点
            # print(tt[i] - slice_pen_times[-1])
            slice_pen_times.append(tt[i+1])
            indexs.append(i+1)
            if time_types[-1] == 2: #如果前一条线也收笔点，则需要修改为折笔
                time_types[-1] = 3
            time_types.append(2)


        if i in min_gap_xy_index:  #第一种折笔
            if tt[i] - slice_pen_times[-1] > time_theshold_changeed:
                slice_pen_times.append(tt[i])
                indexs.append(i)
                time_types.append(3)

        if x * xs[i+1] < 0 or y * ys[i+1] < 0:  #第二种折笔: 过零点
            if tt[i] - slice_pen_times[-1] > time_theshold_changeed:
                slice_pen_times.append(tt[i])
                indexs.append(i)
                time_types.append(3)

    for i in range(len(indexs)-1):
        start_index = indexs[i]
        end_index = indexs[i+1]
        if start_index == end_index: #如果起点和终点相同
            continue
        x_sum = np.abs(np.sum(xs[start_index:end_index]))
        y_sum = np.abs(np.sum(ys[start_index:end_index]))
        x_num = [1 for i in range(start_index,end_index) if np.abs(xs[i]) > 50]  #大于50的个数
        y_num = [1 for i in range(start_index, end_index) if np.abs(ys[i]) > 50]  #大于50的个数
        x_mean = np.mean(xs[start_index:end_index])
        y_mean = np.mean(ys[start_index:end_index])
        x_direction = 2 if x_mean > 0 else 1  # x 轴向右是大于500即大于0， 向左为1，向右为2
        y_direction = 2 if y_mean > 0 else 1  # y 轴向下是大于500即大于0， 向上为1，向下为2

        if time_types[i] == 2: #如果为收笔
            slice_pen_types.append(0)  # 收笔
        elif (x_sum - y_sum) > x_sum * rate_theshold:
            slice_pen_types.append(1) # 横笔
        elif (y_sum - x_sum) > y_sum * rate_theshold:  # 竖笔中有可能是撇笔
            if len(x_num) > 1:
                slice_pen_types.append(3)  # 撇笔
            else:
                slice_pen_types.append(2) # 竖笔
        elif x_sum > y_sum:
            # slice_pen_types.append(31) # 斜笔，第二备选为横笔
            slice_pen_types.append(3)  # 斜笔
        else:
            # slice_pen_types.append(32) # 斜笔，第二备选为竖笔
            slice_pen_types.append(3)  # 斜笔

        if slice_pen_types[-1] != 0:
            slice_pen_types[-1] = slice_pen_types[-1] * 100 + x_direction * 10 + y_direction

    slice_pen_types.append(0) # 最后一笔肯定为收笔


    slice_pen_types_tmp, slice_pen_times_tmp, time_types_tmp, indexs_tmp = [],[],[],[]
    for i in range(len(slice_pen_types)):
        if int(slice_pen_types[i]/100) == 2 and int(slice_pen_types[i+1]/100) == 3 and time_types[i+1] == 3:  # 如果为“竖、撇”的整合成“撇”
            print("================竖、撇”")
            continue
        if i > 0 and int(slice_pen_types[i-1]/100) == int(slice_pen_types[i]/100) and time_types[i] == 3:  # 如果当前笔和前一笔相同，且为折笔，则合并到前一笔
            print("================相同笔”")
            continue
        else:
            slice_pen_types_tmp.append(slice_pen_types[i])
            slice_pen_times_tmp.append(slice_pen_times[i])
            time_types_tmp.append(time_types[i])
            indexs_tmp.append(indexs[i])
    # slice_pen_types_tmp.append(slice_pen_types[-1])
    # slice_pen_times_tmp.append(slice_pen_times[-1])
    # time_types_tmp.append(time_types[-1])
    # indexs_tmp.append(indexs[-1])
    slice_pen_types, slice_pen_times, time_types, indexs = slice_pen_types_tmp,slice_pen_times_tmp,time_types_tmp,indexs_tmp

    return slice_pen_types,slice_pen_times,time_types,indexs

'''
获取每个中文单词的笔划数
'''
def get_stroke_order_samples_from_txt(txt_path):
    stroke_order_samples = []
    with open(txt_path, 'r', encoding='gbk') as fr:
        index = 0
        for line in fr:
            line = line.replace("\n","")
            if index == 0:
                char = line
            elif index == 1:
                xs = line
            elif index == 2:
                ys = line
            elif index == 3:
                start_indexs = line
            elif index == 4:
                end_indexs = line
                sample = StrokeOrderSample()
                test = sample.make_struct(char, xs, ys, start_indexs, end_indexs, None)
                print(test.char)
                stroke_order_samples.append(test)
            index += 1
            index = index%5
    return stroke_order_samples


class StrokeOrderSample(object):
    class Struct(object):
        def __init__(self, char, xs, ys, start_indexs, end_indexs, change_indexs):
            self.char = char
            self.xs = xs
            self.ys = ys
            self.start_indexs = start_indexs
            self.end_indexs = end_indexs
            self.change_indexs = change_indexs

    def make_struct(self, cahr, xs, ys, start_indexs, end_indexs, change_indexs):
        return self.Struct(cahr, xs, ys, start_indexs, end_indexs, change_indexs)

class LevenshteinNearly(object):
    class Struct(object):
        def __init__(self, char, level_zero,level_one,level_two,level_three):
            self.char = char
            self.level_zero = level_zero
            self.level_one = level_one
            self.level_two = level_two
            self.level_three = level_three

    def make_struct(self, cahr, level_zero,level_one,level_two,level_three):
        return self.Struct(cahr, level_zero,level_one,level_two,level_three)

def save_stroke_order_nearly():
    level_zero,level_one,level_two,level_three = [],[],[],[]
    all_stroke_orders = get_all_stroke_orders("stroke-order.txt")
    result = {}
    index = 0
    for k1 in all_stroke_orders.keys():
        s = all_stroke_orders[k1]
        for k2 in all_stroke_orders.keys():
            a = all_stroke_orders[k2]
            dist = Levenshtein.distance(s, a)
            if dist == 0:
                level_zero.append(k2)
            elif dist == 1:
                level_one.append(k2)
            elif dist == 2:
                level_two.append(k2)
            elif dist == 3:
                level_three.append(k2)
        le = LevenshteinNearly()
        test = le.make_struct(k1,level_zero,level_one,level_two,level_three)
        result[s] = test
        level_zero, level_one, level_two, level_three = [], [], [], []
        print(index)
        index += 1
    print('saving result...')
    np.save('stroke-order-nearly', result)

def save_stroke_order_length():
    all_stroke_orders = get_all_stroke_orders("stroke-order.txt")
    result = {}
    index = 0
    for k1 in all_stroke_orders.keys():
        s = all_stroke_orders[k1]
        length = len(s)
        if length in result.keys():
            char_list = result[length]
            char_list.append(k1)
        else:
            char_list = [k1]
            result[length] = char_list
        print(index)
        index += 1
    print('saving result...')
    np.save('stroke-order-length-dirct', result)

def get_nearly_N(before_char,reccent_stroke_order,all_stroke_orders,stroke_order_nearly,stroke_order_length,wordCount,high_info,N=15):
    result = []
    s = reccent_stroke_order
    lenght = len(s)
    if s in stroke_order_nearly.keys() and False:
        test = stroke_order_nearly[s]
        if before_char is None:
            all_found_chars = test.level_zero + test.level_one + test.level_two + test.level_three
        else:
            all_found_chars = order_by_workCount_for_one_char(before_char,reccent_stroke_order, wordCount, stroke_order_nearly)
            if len(all_found_chars) == 0:
                all_found_chars = test.level_zero + test.level_one + test.level_two + test.level_three
        all_found_chars = [a for a in all_found_chars if len(all_stroke_orders[a]) == lenght]
        # print(all_found_chars)
        for a in all_found_chars:
            if a != "" and len(result) < N:
                result.append(a)
            if len(result) == N:
                break
    else:  # 如果不在字典里面，则查找距离为1的相近字符
        length = len(s)
        char_list = []
        if length in stroke_order_length.keys():
            char_list_tmp = stroke_order_length[length]
            char_list += char_list_tmp
        # if length-1 in stroke_order_length.keys():
        #     char_list_tmp = stroke_order_length[length-1]
        #     char_list += char_list_tmp
        # if length+1 in stroke_order_length.keys():
        #     char_list_tmp = stroke_order_length[length+1]
        #     char_list += char_list_tmp

        char_dist = []
        for c in char_list:
            s2 = all_stroke_orders[c]
            dist = Levenshtein.distance(s, s2)
            char_dist.append(dist)
        # 按列表char_dist中元素的值进行排序，并返回元素对应索引序列
        sorted_id = sorted(range(len(char_dist)), key=lambda k: char_dist[k])
        result = [char_list[i] for i in sorted_id]
        stroke_orders = [all_stroke_orders[i] for i in result]
        # high_info = {0:'4',2:'1',4:'3',5:'4',8:'1'}
        found_stroke_orders = [i for i,so in enumerate(stroke_orders) if check_by_high_info(so,high_info)]
        found_chars = [result[i] for i in found_stroke_orders]
        print("found_chars is {}".format(found_chars))
        sorted_dist = [char_dist[i] for i in sorted_id]
        print("sorted_dist is {}".format(sorted_dist[:N]))
        if len(found_chars) > 0:
            if before_char is not None:
                wordCount_detail = wordCount[before_char]
                wordCount_detail_keys = list(wordCount_detail.keys())  # 根据词频找出来的所有联想词
                found_wordCount_indexs = [wordCount_detail_keys.index(l) for l in found_chars if l in wordCount_detail_keys]
                # found_wordCount_indexs = [wordCount_detail_keys.index(l) for l in result if l in wordCount_detail_keys]
                if len(found_wordCount_indexs) > 0:
                    found_wordCount_indexs = sorted(found_wordCount_indexs)
                    found_chars = [wordCount_detail_keys[i] for i in found_wordCount_indexs]
            result = found_chars + [i for i in result if i not in found_chars]
        if len(result) >= N:
            return result[:N]

    return result

def check_by_high_info(stroke_order,high_info):
    flag = 0
    for key in high_info.keys():
        if stroke_order[key] == high_info[key]:
            flag += 1
    if flag == len(high_info):
        return True
    else:
        return False

'''
构造样本的特征值：（start_time,end_time,duration,pen_down,pen_up,stroke_type,stroke_total）(超始时间点、结束时间点、距离下一采样点的时长、起点是否为落笔、起点是否为收笔、笔画种类，笔画总数)
sample_strokes 笔画采样点数据,
stroke_total 笔画总数,
total_duration  规范化的总时长
'''
def time_stroke_order_features_by_times(point_times,total_duration,stroke_order):
    mat_tmp = np.ndarray(shape=(1,), dtype=object)
    stroke_rep = [[]]
    stroke_index = 0
    stroke_total = len(stroke_order)
    rate = total_duration /(point_times[-1] - point_times[0])
    point_times_base = [round(rate * (p - point_times[0]),3) for p in point_times]
    for n, stroke_type in enumerate(stroke_order):
        index = n * 2
        start_time = point_times_base[index]
        end_time = point_times_base[index+1]
        duration = end_time - start_time
        pen_down,pen_up = 1,1
        stroke_type = stroke_order[stroke_index]
        stroke_index += 1
        tmp = [[start_time, end_time, duration, pen_down, pen_up,stroke_type,stroke_total]]
        stroke_rep += tmp
    mat_tmp[0] = stroke_rep[1:]
    return mat_tmp

def save_write_detail(true_char,char,x_tmp,y_tmp,start_indexs,stroke_types):
    # print(list(x_tmp))
    # print(list(y_tmp))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签 https://blog.csdn.net/weixin_41767802/article/details/108047350
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.title(true_char + '-'+ char, fontsize='xx-large', fontweight='heavy')
    plt.xlabel("stroke_num is {}".format(len(start_indexs)))
    plt.ylabel(true_char + '-'+ char)
    plt.plot(x_tmp, '-r*', label='xs %')
    plt.plot(y_tmp, '-g*', label='ys %')
    plt.vlines(start_indexs, 0, 400, 'r', ':')
    plt.legend(loc='upper right', fontsize=10)  # 标签位置
    for i,s in enumerate(start_indexs):
        plt.text(s,400,stroke_types[i])
    plt.savefig("./tmp/detail/" + true_char + '-'+ char + '-' + str(np.round(time.time(), 2)) + ".png")
    plt.close()
    sample_save_path = "./tmp/detail/" + true_char + '-'+ char + '-' + str(np.round(time.time(), 2)) + ".txt"
    with open(sample_save_path, mode='a') as filename:
        xs = ' '.join(str(i).replace("\n", '') for i in x_tmp)
        filename.write(xs)
        filename.write('\n')  # 换行
        ys = ' '.join(str(i).replace("\n", '') for i in y_tmp)
        filename.write(ys)
        ss = ' '.join(str(i).replace("\n", '') for i in start_indexs)
        filename.write(ss)
    # plt.show()

def save_stroke_detail_by_type(txt_path):
    # txt_path = './tmp/samples-zhy.txt'
    stroke_order_samples = get_stroke_order_samples_from_txt(txt_path)
    all_stroke_orders = get_all_stroke_orders("stroke-order.txt")

    data_labels = []
    data_xs = []
    data_ys = []
    for i in range(len(stroke_order_samples)):
        char = stroke_order_samples[i].char
        print(char)
        if char == "川":
            pass
        stroke_order = all_stroke_orders[char]
        stroke_order = [int(s) for s in stroke_order]

        x_tmp = stroke_order_samples[i].xs.split(" ")
        x_tmp = [int(float(i)) for i in x_tmp]

        y_tmp = stroke_order_samples[i].ys.split(" ")
        y_tmp = [int(float(i)) for i in y_tmp]

        start_indexs = stroke_order_samples[i].start_indexs.split(" ")
        start_indexs = [int(float(i)) for i in start_indexs]

        end_indexs = stroke_order_samples[i].end_indexs.split(" ")
        end_indexs = [int(float(i)) for i in end_indexs]

        if len(stroke_order) != len(start_indexs):  # 如果采样的笔划数与标准笔划数不一致
            print("{}: stroke order is not right".format(char))
            continue

        for n in range(len(start_indexs)):  # 组装每一笔的采样数据及标签
            start = start_indexs[n]
            end = end_indexs[n]
            xs = x_tmp[start:end + 1]
            ys = y_tmp[start:end + 1]
            data_xs.append(xs)
            data_ys.append(ys)
            data_labels.append(stroke_order[n])

            stroke_types = "横" if stroke_order[n] == 1 else "竖" if stroke_order[n] == 2 else "撇" if stroke_order[n] == 3 else "捺" if stroke_order[n] == 4 else "折" if stroke_order[n] == 5 else "N"
            save_dirct = str(stroke_order[n])
            save_filename = char + '-' + str(n+1)
            png_save_path = "./tmp/stroke_types/" + save_dirct + '/' + save_filename + '-'+ str(np.round(time.time(), 2)) + ".png"
            sample_save_path = "./tmp/stroke_types/" + save_dirct + '/' + save_filename + '-'+ str(np.round(time.time(), 2)) + ".txt"

            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签 https://blog.csdn.net/weixin_41767802/article/details/108047350
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
            plt.title(char + '-' + str(n+1) + '-' + stroke_types, fontsize='xx-large', fontweight='heavy')
            plt.xlabel("stroke_num is {}".format(len(start_indexs)))
            plt.ylabel(char + '-' + str(n+1) + '-' + stroke_types)
            plt.plot(xs, '-r*', label='xs %')
            plt.plot(ys, '-g*', label='ys %')
            plt.legend(loc='upper right', fontsize=10)  # 标签位置
            plt.savefig(png_save_path)
            plt.close()
            with open(sample_save_path, mode='a') as filename:
                xs = ' '.join(str(i).replace("\n", '') for i in xs)
                filename.write(xs)
                filename.write('\n')  # 换行
                ys = ' '.join(str(i).replace("\n", '') for i in ys)
                filename.write(ys)

def save_wordCount():
    result = {}
    for root, dirs, files in os.walk('E:/ocr_data/database/yuliao/复旦分类语料/answer/'):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list
        files = [f for f in files if f.find('txt') >= 0]

        # 遍历文件
        for f in files:
            if f.find('png') >= 0:
                continue
            txt_path = os.path.join(root, f)
            print(txt_path)
            article = open(txt_path, 'r',encoding='gb18030',errors='ignore').read()
            # article = bytes(article,encoding='utf8')
            article = del_all_punctuation(article)
            # print(article)
            for i in range(len(article) - 1):
                c = article[i]
                n = article[i + 1]
                if c in result.keys():
                    detail = result[c]
                    if n in detail.keys():
                        detail[n] += 1
                    else:
                        detail[n] = 1
                else:
                    detail = {}
                    detail[n] = 1
                    result[c] = detail
    for key in result.keys():
        result[key] = sorted_dict(result[key])
    print('saving result...')
    np.save('E:/ocr_data/model/wordCount', result)

def sorted_dict(d):
    # 第一种方法，key使用lambda匿名函数取value进行排序
    # a = sorted(d.items(), key=lambda x: x[1])
    a1 = sorted(d.items(), key=lambda x: x[1], reverse=True)
    # print(a)
    # print(a1)
    names = [letter[0] for letter in a1]
    values = [letter[1] for letter in a1]
    result = dict(zip(names, values))
    return result

def del_all_punctuation(text):
    en_punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for i in en_punctuation:
        text = text.replace(i, '')

    cn_punctuation = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
    for i in cn_punctuation:
        text = text.replace(i, '')

    # 去掉数字
    text = re.sub('[\d]','',text) # [0-9]

    # 去除英文
    text = re.sub('[a-zA-Z]', '', text)

    # 去除空格
    text = re.sub('[\s]', '', text)  # temp = text.strip()

    return text

def order_by_workCount_for_one_char(befor_char,reccent_stroke_order,wordCount,stroke_order_nearly):
    detail = wordCount[befor_char]
    detail_keys = list(detail.keys())  # 根据词频找出来的所有联想词
    # s = all_stroke_orders[char]
    test = stroke_order_nearly[reccent_stroke_order]  # 所有的笔画临近的词
    level_zero = test.level_zero  # 所有相同笔画的词
    level_zero_indexs = [detail_keys.index(l) for l in level_zero if l in detail_keys]
    level_zero_indexs = sorted(level_zero_indexs)
    level_one = test.level_one  # 所有差1笔的词
    level_one_indexs = [detail_keys.index(l) for l in level_one if l in detail_keys]
    level_one_indexs = sorted(level_one_indexs)
    level_two = test.level_two  # 所有差2笔的词
    level_two_indexs = [detail_keys.index(l) for l in level_two if l in detail_keys]
    level_two_indexs = sorted(level_two_indexs)
    level_three = test.level_three  # 所有差3笔的词
    level_three_indexs = [detail_keys.index(l) for l in level_three if l in detail_keys]
    level_three_indexs = sorted(level_three_indexs)
    all_indexs = level_zero_indexs + level_one_indexs + level_two_indexs + level_three_indexs
    # print(all_indexs)
    all_found_chars = [detail_keys[i] for i in all_indexs]
    return all_found_chars

def order_by_workCount_for_chars(befor_char,level_chars,wordCount):
    detail = wordCount[befor_char]
    detail_keys = list(detail.keys())  # 根据词频找出来的所有联想词
    level_indexs = [detail_keys.index(l) for l in level_chars if l in detail_keys]
    level_indexs = sorted(level_indexs)
    # print(level_indexs)
    all_found_chars = [detail_keys[i] for i in level_indexs]
    return all_found_chars

'''
坐标绽放
'''
def make_scale(original, width_scaleFactor, height_scaleFactor):
    scaled = copy.deepcopy(original)

    for i in range(len(scaled)):
        scaled[i] = (int(scaled[i][0] * width_scaleFactor), int(scaled[i][1] * height_scaleFactor),scaled[i][2])
    return scaled

'''
坐标平移
'''
def make_move(original,x_offset,y_offset,x_pad=0,y_pad=0):
    scaled = copy.deepcopy(original)

    for i in range(len(scaled)):
        scaled[i] = (int(scaled[i][0] + x_offset + x_pad), int(scaled[i][1] + y_offset + y_pad),scaled[i][2])
    return scaled

'''
初始化画板
'''
def init_canvas(width,height,row_number,col_number):
    img = 255 * np.ones([height, width, 3], dtype=np.uint8)
    row_gap = height/row_number
    col_gap = width/col_number
    for i in range(1,row_number):
        tmp = int(row_gap * i)
        cv2.line(img, (10, tmp), (width - 10, tmp), (0, 0, 255), 1)
    cells = np.ndarray(shape=(row_number,col_number), dtype=object)
    for i in range(row_number):
        for j in range(col_number):
            x = int(col_gap * j)
            y = int(row_gap * i)
            cells[i,j] = (x,y)
    return img,cells

'''
将原点处的坐标平移到指定位置
'''
def moved_zero_by_position(cells,row_index,col_index,original_in_zero,x_offset=3,y_offset=3):
    x, y = cells[row_index,col_index]
    moved = make_move(original_in_zero, x + x_offset, y + y_offset)
    return moved

'''
将原始坐标平移到指定位置
'''
def moved_original_by_position(cells,row_gap,col_gap,row_index,col_index,original,x_offset=5,y_offset=3):
    x_min, y_min = np.min([x[0] for x in original]), np.min([y[1] for y in original])
    x_max, y_max = np.max([x[0] for x in original]), np.max([y[1] for y in original])
    original_width = x_max - x_min
    original_height = y_max - y_min
    row_scaleFactor = row_gap / original_height * 0.8
    col_scaleFactor = col_gap / original_width * 0.8
    scaled = make_scale(original, col_scaleFactor, row_scaleFactor)

    x_min, y_min = np.min([x[0] for x in scaled]), np.min([y[1] for y in scaled])
    original_in_zero = make_move(scaled, -x_min, -y_min)

    x_offset, y_offset = col_gap * 0.1, row_gap * 0.1
    result = moved_zero_by_position(cells, row_index, col_index, original_in_zero,x_offset,y_offset)
    return result

'''
画出笔划轨迹
'''
def draw(img,item,lineWidth=7):
    for i in range(1, len(item) - 2):
        # print(i)
        start = item[i]
        end = item[i - 1]
        # print(start)
        if start[-1] == 1:
            start = (start[0], start[1])
            end = (end[0], end[1])
            # cv2.circle(img, start, 3, (0,0,0), -1)
            # cv2.circle(img, end, 3, (0,0,0), -1)
            cv2.line(img, start, end, (0, 0, 0), lineWidth)

'''

'''
def get_draw_position(cells,total):
    row_number,col_number = cells.shape
    row_index = math.floor(total/col_number)
    col_index = total % col_number

    if row_index >= row_number:
        return -1,-1
    return row_index,col_index