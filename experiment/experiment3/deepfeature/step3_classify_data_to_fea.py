import os
import numpy as np
import re

def divide_pic_to_fea(adv_list, save_path, load_path):
    for adv_name in adv_list:
        save_adv_path = os.path.join(save_path, adv_name)
        if not os.path.exists(save_adv_path):
            os.makedirs(save_adv_path)
        load_adv_path = os.path.join(load_path, adv_name, 'prediction')
        for class_num in range(10):
            predict_x = np.load(os.path.join(load_adv_path, str(class_num) + '_x.npy'))
            predict_name = np.load(os.path.join(load_adv_path, str(class_num) + '_name.npy'))
            predict_fea_name = np.load(os.path.join(load_adv_path, str(class_num) + '_fea_name.npy'))
            score = np.load(os.path.join(load_adv_path, str(class_num) + '_fea_score.npy'))
            fea_name_arr = np.unique(predict_fea_name)
            fea_pic_x = []
            fea_pic_name = []
            fea_pic_score = []
            for fea_num in range(len(fea_name_arr)):
                fea_pic_x.append([])
                fea_pic_name.append([])
                fea_pic_score.append([])
            for pic_num in range(len(predict_x)):
                index = np.where(fea_name_arr == predict_fea_name[pic_num])[0][0]
                fea_pic_x[index].append(predict_x[pic_num])
                fea_pic_name[index].append(predict_name[pic_num])
                fea_pic_score[index].append(score[pic_num])
            fea_pic_x = np.array(fea_pic_x)
            fea_pic_name = np.array(fea_pic_name)
            np.save(os.path.join(save_adv_path, str(class_num) + '_fea_pic_x.npy'), fea_pic_x)
            np.save(os.path.join(save_adv_path, str(class_num) + '_fea_pic_name.npy'), fea_pic_name)
            np.save(os.path.join(save_adv_path, str(class_num) + '_fea_name.npy'), fea_name_arr)
            np.save(os.path.join(save_adv_path, str(class_num) + '_fea_score.npy'), fea_pic_score)
            print(len(fea_pic_x[0]))
            print(len(fea_pic_name[0]))
            print(fea_name_arr)


def cal_region_acc(adv_list, save_path):
    char = '_'
    for adv_name in adv_list:
        for class_num in range(10):
            acc = []
            class_fea_pic_name = os.path.join(save_path, adv_name, str(class_num) + '_fea_pic_name.npy')
            fea_pic_name = np.load(class_fea_pic_name, allow_pickle=True)
            for fea_num in range(len(fea_pic_name)):
                pic_name = fea_pic_name[fea_num]
                tru_num = 0
                fal_num = 0
                # print(pic_name)
                for name in pic_name:
                    truth_class_name = re.split(re.escape(char), name)[0]
                    # print(name)
                    # print(truth_class_name)
                    if truth_class_name == str(class_num):
                        tru_num += 1
                    else:
                        fal_num += 1
                temp_acc = float(tru_num) / float(tru_num + fal_num)
                acc.append(temp_acc)
            acc = np.array(acc)
            print(acc)
            np.save(os.path.join(save_path, adv_name, str(class_num) + '_fea_acc.npy'), acc)
if __name__ == '__main__':
    pass