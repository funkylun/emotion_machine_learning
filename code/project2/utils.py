import codecs
import csv
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
import cv2
import scipy.signal as signal  # pip install scipy==1.1.0
import torch
from torchvision.transforms import ToTensor

def export_csv(datas,save_filename):

    try:
        # 1. 创建文件对象
        f = codecs.open(save_filename,'wb', "gbk")
        # f.write(codecs.BOM_UTF8)  # 防止乱码

        # 2. 基于文件对象构建 csv写入对象
        csv_writer = csv.writer(f)

        # 3. 构建列表头
        csv_writer.writerow(["frame_index", "disp_arousal", "disp_valence","disp_intensity"])

        # 4. 写入多组数据存放list列表里面csv文件内容
        # tmp = []
        # for i,d in enumerate(datas):
        #     t = list(d)
        #     # csv_writer.writerow(d)

        csv_writer.writerows(datas)

    except Exception as e:
        print(e)
    finally:
        # 5. 关闭文件
        f.close()

'''
X为二维数据，shape=(3,N), 
'arousal ', 'valence', 'intensity'
'''
def plot_subject(X):
    '''
    该函数实现根据志愿者标号绘制其9轴传感器数据和活动变化图
    共十个子图
    '''
    plt.figure(figsize=(8, 10), dpi=80)
    n, off = X.shape[0], 0  # X的第三个维度为特征数9（9轴传感器数据）

    name_list = ['arousal ', 'valence', 'intensity']
    for i, name in enumerate(name_list, start=0):
        plt.subplot(n, 1, off + 1)  # 创建n行1列的画布，在off+1位置绘图；
        # X[:,:,off] 三维数组切片中，off通过for循环实现递增，
        # 一次截取一个特征的所有数据（二维数组），输入到to_series函数进行处理，实现去除重叠部分。
        plt.plot(X[i,:])
        plt.title(name, y=0, loc='right', size=12)
        plt.ylabel(r'$Value$', size=12)
        plt.xlabel(r'$timesteps$', size=14)
        off += 1

    plt.tight_layout()
    plt.show()

'''
X为二维数据，shape=(3,N), 
'arousal ', 'valence', 'intensity'
'''
def plot_subject_compare(X1,X2):
    '''
    对比单一个体'arousal ', 'valence', 'intensity'的数据
    '''
    plt.figure(figsize=(8, 12), dpi=70)
    n, off = X1.shape[0], 0  # X的第三个维度为特征数9（9轴传感器数据）

    name_list = ['arousal ', 'valence', 'intensity']
    for i, name in enumerate(name_list, start=0):
        plt.subplot(n, 2, off + 1 )  # 创建n行2列的画布，在off+1位置绘图；
        plt.plot(X1[i,:])
        plt.title(name, y=0, loc='right', size=12)
        plt.ylabel(name_list[i], size=12)
        plt.xlabel(r'$timesteps$', size=14)
        # off += 1
        plt.subplot(n, 2, off + 2)  # 创建n行2列的画布，在off+1位置绘图；
        plt.plot(X2[i, :])
        plt.title(name, y=0, loc='right', size=12)
        plt.ylabel(name_list[i], size=12)
        plt.xlabel(r'$timesteps$', size=14)
        off += 2

    plt.tight_layout()
    plt.show()

'''
X为三维数据，shape=(N,3,F), 
'arousal ', 'valence', 'intensity'
'''
def plot_all_subjects_compare(X1,X2):
    '''
    对比所有个体'arousal ', 'valence', 'intensity'的数据
    '''
    plt.figure(figsize=(8, 10), dpi=80)
    n, off = X1.shape[1], 0  # X的第三个维度为特征数9（9轴传感器数据）
    zhfont = FontProperties(fname='../assets/simsun.ttc')

    name_list = ['arousal ', 'valence', 'intensity']
    for i, name in enumerate(name_list, start=0):
        plt.subplot(n, 2, off + 1 )  # 创建n行2列的画布，在off+1位置绘图；
        for j in range(X1.shape[0]):
            plt.plot(X1[j,i])
        plt.title('正常人', y=0, loc='right', size=12,fontproperties=zhfont)
        plt.ylabel(name_list[i], size=12)
        plt.xlabel('timesteps', size=14)
        # off += 1
        plt.subplot(n, 2, off + 2)  # 创建n行2列的画布，在off+1位置绘图；
        for j in range(X2.shape[0]):
            plt.plot(X2[j,i])
        plt.title('痴呆患者', y=0, loc='right', size=12,fontproperties=zhfont)
        plt.ylabel(name_list[i], size=12)
        plt.xlabel('timesteps', size=14)
        off += 2

    plt.tight_layout()
    plt.show()

def get_data_from_csv(path,type):
    emotions = []
    smaple_flag = True
    # smaple_flag = False
    sample_step = 500
    filterd = True
    medfilt_flag = True
    for home, dirs, files in os.walk(path):
        # 获取xlsx文件信息
        if len(files) > 0:
            filenames = [filename for filename in files if filename.find('csv') > 0]

        for fs in filenames:
            details = fs.split("-")
            # person_id = details[0] + "-" + details[1]
            file_type = details[-2]
            if type == file_type:
                filename = os.path.join(home,fs)
                print(filename)
                data = pd.read_csv(filename, encoding='ANSI')
                arousal = list(data['disp_arousal'].values)
                valence = list(data['disp_valence'].values)
                intensity = list(data['disp_intensity'].values)

                if filterd:
                    zero_positions = [i for i in range(len(arousal)) if arousal[i] == 0.0 and valence[i] == 0.0 and intensity[i] == 0]
                    arousal = [arousal[i] for i in range(len(arousal)) if i not in zero_positions]
                    valence = [valence[i] for i in range(len(valence)) if i not in zero_positions]
                    intensity = [intensity[i] for i in range(len(intensity)) if i not in zero_positions]
                # arousal = [arousal[i] if arousal[i] != 0 else arousal[i-1] for i in range(1,len(arousal)-1)]
                # valence = [valence[i] if valence[i] != 0 else valence[i - 1] for i in range(1, len(valence) - 1)]
                # intensity = [intensity[i] if intensity[i] != 0 else intensity[i - 1] for i in range(1, len(intensity) - 1)]
                # if medfilt_flag:
                #     arousal = signal.medfilt(arousal, 15)
                #     valence = signal.medfilt(valence, 153)
                #     intensity = signal.medfilt(intensity, 15)
                    # arousal = signal.savgol_filter(arousal, 53,3)

                if sample_step > len(arousal):
                    continue

                if smaple_flag:
                    arousal_sample_positions = np.linspace(0, len(arousal), sample_step, endpoint=False)
                    arousal_sample_positions = [int(s) for s in arousal_sample_positions]
                    arousal = [arousal[i] for i in arousal_sample_positions]
                    valence_sample_positions = np.linspace(0, len(valence), sample_step, endpoint=False)
                    valence_sample_positions = [int(s) for s in valence_sample_positions]
                    valence = [valence[i] for i in valence_sample_positions]
                    intensity_sample_positions = np.linspace(0, len(intensity), sample_step, endpoint=False)
                    intensity_sample_positions = [int(s) for s in intensity_sample_positions]
                    intensity = [intensity[i] for i in intensity_sample_positions]
                tmp = [arousal,valence,intensity]
                emotions.append(tmp)
            # if len(emotions) > 15:
            #     break
    # all_mats = np.ndarray(shape=(len(emotions),), dtype=object)
    # for i in range(len(emotions)):
    #     all_mats[i] = emotions[i]
    emotions = np.array(emotions)
    return emotions

'''
X为三维数据，shape=(N,3,F), 
'arousal ', 'valence', 'intensity'
'''
def plot_arousal_and_valence_all_subjects_compare(X1,X2,video_type):

    video_dict = {"01":"悲伤", "02":"厌恶", "03":"愤怒", "04":"恐惧", "05":"中性", "06":"开心"}
    type = video_dict[video_type]

    plt.figure(figsize=(16, 8), dpi=80)
    n, off = X1.shape[1], 0  # X的第三个维度为特征数9（9轴传感器数据）
    zhfont = FontProperties(fname='../assets/simsun.ttc')
    plt.suptitle("视频类型为" + type,size=20,fontproperties=zhfont)

    plt.subplot(1, 2, 1)  # 创建n行2列的画布，在off+1位置绘图；
    # plt.title('正常人', y=0, loc='right', size=12, fontproperties=zhfont)
    plt.title('Arousal Valence Space 正常人' + "(样本数：" + str(X1.shape[0]) +"人)",size=16,fontproperties=zhfont)
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.axhline(linewidth=3, color='k')
    plt.axvline(linewidth=3, color='k')
    plt.grid(True)

    for j in range(X1.shape[0]):
        plt.plot(X1[j][0], X1[j][1],marker='.',markersize=4)

    plt.subplot(1, 2, 2)  # 创建n行2列的画布，在off+1位置绘图；
    plt.title('Arousal Valence Space 痴呆患者' + "(样本数：" + str(X2.shape[0]) +"人)",size=16,fontproperties=zhfont)
    plt.xlim((-1.5, 1.5))
    plt.ylim((-1.5, 1.5))
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.axhline(linewidth=3, color='k')
    plt.axvline(linewidth=3, color='k')
    plt.grid(True)

    for j in range(X2.shape[0]):
        plt.plot(X2[j][0], X2[j][1], marker='.', markersize=4)


    plt.tight_layout()
    plt.show()

def check_normal_data():
    normal_path = 'E:/mer_data/code/facial-expression-analysis-main/source/output/normal/'
    for home, dirs, files in os.walk(normal_path):
        # 获取xlsx文件信息
        if len(files) > 0:
            normal_person_ids = ["-".join(filename.split("-")[:-2]) for filename in files if filename.find('csv') > 0]
    normal_person_ids = set(normal_person_ids)

    normal_path = 'G:/mer-data/第三次（70正常+70痴呆）/正常/'
    for home, dirs, files in os.walk(normal_path):
        # 获取xlsx文件信息
        if len(dirs) > 0:
            all_normal_person_ids = dirs
            break

    normal_path = 'G:/mer-data/第一次（27正常+39痴呆）/正常/'
    for home, dirs, files in os.walk(normal_path):
        # 获取xlsx文件信息
        if len(dirs) > 0:
            all_normal_person_ids += dirs
            break

    all_normal_person_ids = set(all_normal_person_ids)
    for np in normal_person_ids:
        if np not in all_normal_person_ids:
            print("error data : {}".format(np))
    # print(all_normal_person_ids)

def check_normal_not_data():
    normal_not_path = 'E:/mer_data/code/facial-expression-analysis-main/source/output/normal_not/'
    for home, dirs, files in os.walk(normal_not_path):
        # 获取xlsx文件信息
        if len(files) > 0:
            person_ids = ["-".join(filename.split("-")[:-2]) for filename in files if filename.find('csv') > 0]
    person_ids = set(person_ids)

    normal_not_path = 'G:/mer-data/第三次（70正常+70痴呆）/痴呆/'
    for home, dirs, files in os.walk(normal_not_path):
        # 获取xlsx文件信息
        if len(dirs) > 0:
            all_person_ids = dirs
            break

    normal_not_path = 'G:/mer-data/第一次（27正常+39痴呆）/痴呆/'
    for home, dirs, files in os.walk(normal_not_path):
        # 获取xlsx文件信息
        if len(dirs) > 0:
            all_person_ids += dirs
            break

    all_person_ids = set(all_person_ids)
    for np in person_ids:
        if np not in all_person_ids:
            print("error data : {}".format(np))
    # print(all_normal_person_ids)

def extract_features(detector,net,fullname,save_filename):
    expressions = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise', 4: 'fear', 5: 'disgust', 6: 'anger',7: 'contempt', 8: 'none'}
    expressions_cn = {0: '中性', 1: '高兴', 2: '悲伤', 3: '惊喜', 4: '恐惧', 5: '厌恶', 6: '愤怒',7: '鄙视', 8: 'none'}
    expressions_indices = {8: [0, 1, 2, 3, 4, 5, 6, 7], 5: [0, 1, 2, 3, 6]}
    frame_index = 0
    frame_count = 0
    EXP_AVG = 0.7  # 70% the new value 30% the old value
    points_av = None
    points_arousal = None
    polyg_arousal = None
    points_valence = None
    polyg_valence = None
    points_intensity = None
    polyg_intensity = None

    ls_arousal = []
    ls_valence = []
    ls_intensity = []

    ls_captures = []

    disp_arousal = 0
    disp_valence = 0
    disp_intensity = 0
    device = 'cpu'
    if fullname:
        cap = cv2.VideoCapture(fullname)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    while True:  # for 300 frames
        print('frame', frame_index)
        if frame_index == 460:
            print("==")

        # TODO: find a better way of closing this window. At the moment it
        # keeps going!
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     print('Q pressed')
        #     cap.release()
        #     cv2.destroyAllWindows()
        #     f = 300
        #     break

        ret, frame = cap.read()  # capture a frame

        if frame is None or ret is not True:
            break
        # if frame_index % 10 != 0:  # 间隔多少帧识别一次
        #     frame_index += 1
        #     continue

        # if frame_index > 100:
        #     break

        frame_index += 1
        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = frame
        faces = detector(image)  # detect faces

        if len(faces) > 0:  # if there are faces detected

            # find larger detected face and select it
            face_size = 0
            idx_largest_face = 0
            if len(faces) >= 1:
                for i, face in enumerate(faces):
                    current_size = ((face.bottom() - face.top()) * (face.right() - face.left()))
                    if face_size < current_size:
                        face_size = current_size
                        idx_largest_face = i

            face = faces[idx_largest_face]  # the face to be processed
            # image_clip = image[face.left():face.right(), face.top():face.bottom()]
            offset = 15
            top = face.top() - offset if face.top() - offset > 0 else 0
            bottom = face.bottom() + offset if face.bottom() + offset < image.shape[0] else image.shape[0]
            left = face.left() - offset if face.left() - offset > 0 else 0
            right = face.right() + offset if face.right() + offset < image.shape[1] else image.shape[1]
            image_clip = image[top:bottom, left:right]
            # cv2.imshow("sss", image_clip)
            if len(image_clip) == 0:
                ls_arousal.append(0)
                ls_valence.append(0)
                ls_intensity.append(0)
                continue
            size = (256, 256)
            images = cv2.resize(image_clip, size, interpolation=cv2.INTER_AREA)
            image_to_tensor = ToTensor()
            y = image_to_tensor(images)
            y = y.unsqueeze(dim=0)
            # print(y.shape)
            y = y.to(device)
            with torch.no_grad():
                out = net(y)
            val = out['valence']
            ar = out['arousal']
            expr = out['expression']
            expr_index = np.argmax(np.squeeze(expr.cpu().numpy()), axis=0)
            expr_name = expressions[expr_index]
            expr_name_cn = expressions_cn[expr_index]
            # inte = out['intensity']
            val = np.squeeze(val.cpu().numpy())
            ar = np.squeeze(ar.cpu().numpy())
            print("expr_name,valence,arousal: {},{},{},{}".format(expr_name,expr_name_cn,val,ar))
            # inte = np.squeeze(inte.cpu().numpy())
            # print(val)
            # print(ar)
            disp_arousal,disp_valence = ar,val

            # append buffer lists
            ls_arousal.append(disp_arousal)
            ls_valence.append(disp_valence)
            ls_intensity.append(disp_intensity)
        else:
            ls_arousal.append(0)
            ls_valence.append(0)
            ls_intensity.append(0)
    all_indexs = list(range(0,len(ls_arousal)-1))
    data = [all_indexs,ls_arousal,ls_valence,ls_intensity]
    data = [[row[i] for row in data] for i in range(len(data[0]))] # 转置
    export_csv(data, save_filename)

def show_features(detector,net,fullname):
    expressions = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'surprise', 4: 'fear', 5: 'disgust', 6: 'anger',7: 'contempt', 8: 'none'}
    expressions_cn = {0: '中性', 1: '高兴', 2: '悲伤', 3: '惊喜', 4: '恐惧', 5: '厌恶', 6: '愤怒',7: '鄙视', 8: 'none'}
    expressions_indices = {8: [0, 1, 2, 3, 4, 5, 6, 7], 5: [0, 1, 2, 3, 6]}
    frame_index = 0

    ls_arousal = []
    ls_valence = []
    ls_intensity = []

    disp_intensity = 0
    device = 'cpu'
    if fullname:
        cap = cv2.VideoCapture(fullname)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    while True:  # for 300 frames
        print('frame', frame_index)
        if frame_index == 460:
            print("==")

        # TODO: find a better way of closing this window. At the moment it
        # keeps going!
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     print('Q pressed')
        #     cap.release()
        #     cv2.destroyAllWindows()
        #     f = 300
        #     break

        ret, frame = cap.read()  # capture a frame

        if frame is None or ret is not True:
            break
        if frame_index % 10 != 0:  # 间隔多少帧识别一次
            frame_index += 1
            continue

        # if frame_index > 100:
        #     break

        frame_index += 1
        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image = frame
        faces = detector(image)  # detect faces

        if len(faces) > 0:  # if there are faces detected

            # find larger detected face and select it
            face_size = 0
            idx_largest_face = 0
            if len(faces) >= 1:
                for i, face in enumerate(faces):
                    current_size = ((face.bottom() - face.top()) * (face.right() - face.left()))
                    if face_size < current_size:
                        face_size = current_size
                        idx_largest_face = i

            face = faces[idx_largest_face]  # the face to be processed
            # image_clip = image[face.left():face.right(), face.top():face.bottom()]
            offset = 15
            top = face.top() - offset if face.top() - offset > 0 else 0
            bottom = face.bottom() + offset if face.bottom() + offset < image.shape[0] else image.shape[0]
            left = face.left() - offset if face.left() - offset > 0 else 0
            right = face.right() + offset if face.right() + offset < image.shape[1] else image.shape[1]
            image_clip = image[top:bottom, left:right]
            # cv2.imshow("sss", image_clip)
            if len(image_clip) == 0:
                ls_arousal.append(0)
                ls_valence.append(0)
                ls_intensity.append(0)
                continue
            size = (256, 256)
            images = cv2.resize(image_clip, size, interpolation=cv2.INTER_AREA)
            image_to_tensor = ToTensor()
            y = image_to_tensor(images)
            y = y.unsqueeze(dim=0)
            # print(y.shape)
            y = y.to(device)
            with torch.no_grad():
                out = net(y)
            val = out['valence']
            ar = out['arousal']
            expr = out['expression']
            expr_index = np.argmax(np.squeeze(expr.cpu().numpy()), axis=0)
            expr_name = expressions[expr_index]
            expr_name_cn = expressions_cn[expr_index]
            # inte = out['intensity']
            val = np.squeeze(val.cpu().numpy())
            ar = np.squeeze(ar.cpu().numpy())
            print("expr_name,valence,arousal: {},{},{},{}".format(expr_name,expr_name_cn,val,ar))
            size = (800, 600)
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            expr_name = "happy"
            label = "expr_name: {}".format(expr_name)
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
            draw_arousal_valence(frame, val, ar)
            cv2.imshow('frame', frame)

def draw_arousal_valence(img,arousal,valence):
    a_width,a_height = 20,150
    v_width,v_height = 30,150
    a_init_point,v_init_point = (20,400),(20,410)
    # img_height,img_width = img.shape
    cv2.rectangle(img, v_init_point, (a_init_point[0] + v_height, a_init_point[1] + v_width), (0, 0, 255))    # 横条
    cv2.rectangle(img, v_init_point, (a_init_point[0] + int(v_height*valence), a_init_point[1] + v_width), (0, 0, 255),thickness=-1)  # 横条填充

    cv2.rectangle(img,a_init_point,(a_init_point[0]+a_width,a_init_point[1]-a_height),(0,0,255))    # 竖条
    cv2.rectangle(img, a_init_point, (a_init_point[0] + a_width, a_init_point[1] - int(a_height*arousal)), (0, 0, 255),thickness=-1)  # 竖条填充