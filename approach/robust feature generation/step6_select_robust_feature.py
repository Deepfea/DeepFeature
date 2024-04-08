import os
import numpy as np

def save_robust_feature(cav_path, score_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for class_num in range(10):
        print(class_num)
        class_cav_path = os.path.join(cav_path, str(class_num) + '_cav.npy')
        class_score_path = os.path.join(score_path, str(class_num) + '_scores.npy')
        fea_name_path = os.path.join(cav_path, str(class_num) + '_fea_name.npy')
        fea_cav = np.load(class_cav_path)
        fea_scores = np.load(class_score_path)
        fea_name = np.load(fea_name_path)
        robust_fea = []
        robust_fea_cav = []
        robust_fea_score = []
        for fea_num in range(len(fea_name)):
            if fea_scores[fea_num] > 0:
                robust_fea.append(fea_name[fea_num])
                robust_fea_cav.append(fea_cav[fea_num])
                robust_fea_score.append(fea_scores[fea_num])
        robust_fea = np.array(robust_fea)
        print(robust_fea)
        robust_fea_cav = np.array(robust_fea_cav)
        print(robust_fea_cav)
        robust_fea_score = np.array(robust_fea_score)
        print(robust_fea_score)
        np.save(os.path.join(save_path, str(class_num) + '_robust_fea.npy'), robust_fea)
        np.save(os.path.join(save_path, str(class_num) + '_robust_fea_cav.npy'), robust_fea_cav)
        np.save(os.path.join(save_path, str(class_num) + '_robust_fea_score.npy'), robust_fea_score)

if __name__ == '__main__':
    pass
