import numpy as np
from keras.models import load_model, Model
import os
from tqdm import tqdm
import math

def select_adv_index(load_score_path, load_deepfeature_path, save_index_path, adv_list, rate):

    if not os.path.exists(save_index_path):
        os.makedirs(save_index_path)
    for adv_name in adv_list:
        print(adv_name)
        save_adv_path = os.path.join(save_index_path, adv_name)
        if not os.path.exists(save_adv_path):
            os.makedirs(save_adv_path)
        adv_score = np.load(os.path.join(load_score_path, adv_name, 'all_adv_score.npy'), allow_pickle=True)

        for class_num in range(10):
            class_adv_score = adv_score[class_num]
            class_adv_score = np.array(class_adv_score)
            sort = class_adv_score.argsort()
            can_adv_index = []
            can_adv_score = []
            for sort_num in range(len(sort)):
                if class_adv_score[sort[sort_num]] > 0:
                    can_adv_index.append(sort[sort_num])
                    can_adv_score.append(class_adv_score[sort[sort_num]])
            can_adv_index = np.array(can_adv_index)
            can_adv_score = np.array(can_adv_score)

            for temp_rate in rate:
                save_rate_path = os.path.join(save_adv_path, str(temp_rate))
                if not os.path.exists(save_rate_path):
                    os.makedirs(save_rate_path)
                load_deepfeature_retrain_path = os.path.join(load_deepfeature_path, adv_name, str(temp_rate),
                                                             str(class_num) + '_y.npy')
                deepfeature_retrain = np.load(load_deepfeature_retrain_path)
                select_num = len(deepfeature_retrain)
                select_index = can_adv_index[:select_num]
                select_score = can_adv_score[:select_num]
                print(len(select_score))
                print(len(select_index))
                np.save(os.path.join(save_rate_path, str(class_num) + '_select_index.npy'), select_index)
                np.save(os.path.join(save_rate_path, str(class_num) + '_select_score.npy'), select_score)

if __name__ == '__main__':
    pass

