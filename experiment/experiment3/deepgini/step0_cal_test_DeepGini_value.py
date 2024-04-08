import numpy as np
from keras.models import load_model, Model
import os
from tqdm import tqdm

def com_DeepGini(model_path, test_path, adv_path, save_path, adv_list, type_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for adv_name in adv_list:
        print(adv_name + ':')
        save_adv_path = os.path.join(save_path, adv_name)
        if not os.path.exists(save_adv_path):
            os.makedirs(save_adv_path)
        all_test_x = []
        for class_num in range(10):
            if type_name == 'test':
                x_test = np.load(os.path.join(test_path, str(class_num) + '_test_x.npy'))
            else:
                x_test = np.load(os.path.join(adv_path, adv_name, str(class_num) + '_all_x.npy'))
            all_test_x.append(x_test)
        all_test_x = np.array(all_test_x)
        all_test_score = []
        model = load_model(model_path)
        for class_num in range(10):
            print('class: ' + str(class_num))
            class_test_score = []
            class_test_x = all_test_x[class_num]
            temp_y = model.predict(class_test_x)
            for num in tqdm(range(len(temp_y))):
                temp = temp_y[num]

                t = float(0)
                for i in range(len(temp)):
                    t = t + temp[i] * temp[i]
                t = 1 - t
                class_test_score.append(t)
            all_test_score.append(class_test_score)
        all_test_score = np.array(all_test_score)
        if type_name == 'test':
            np.save(os.path.join(save_adv_path, 'all_test_score.npy'), all_test_score)
        else:
            np.save(os.path.join(save_adv_path, 'all_adv_score.npy'), all_test_score)

if __name__ == '__main__':
    pass




