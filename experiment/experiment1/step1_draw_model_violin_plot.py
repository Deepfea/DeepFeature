import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

def make_fea_dis(load_path):
    df = pd.DataFrame(columns=('type', 'value', 'hue'))  # 生成空的pandas表
    for class_num in range(10):
        print('class_num: ' + str(class_num))
        class_in_dis = np.load(os.path.join(load_path, 'in_dis', str(class_num) + '_in_dis.npy'), allow_pickle=True)
        class_ou_dis = np.load(os.path.join(load_path, 'ou_dis', str(class_num) + '_ou_dis.npy'), allow_pickle=True)
        for fea_num in range(len(class_in_dis)):
            print('fea_num: ' + str(fea_num))
            fea_in_dis = class_in_dis[fea_num]
            fea_in_dis = np.array(fea_in_dis)
            print(fea_in_dis)
            result = fea_in_dis[np.where((fea_in_dis < 0.5))]
            print('sum:' + str(len(fea_in_dis)))
            print(len(result))
            temp = np.random.randint(0, len(fea_in_dis), 10)
            for dis_num in tqdm(range(10)):
                temp_type = str(class_num) + '_' + str(fea_num)
                df = df.append({'Feature': temp_type, 'Distance': fea_in_dis[temp[dis_num]], 'Type': 'inner'}, ignore_index=True)
        for fea_num in range(len(class_ou_dis)):
            print('fea_num: ' + str(fea_num))
            fea_ou_dis = class_ou_dis[fea_num]
            fea_ou_dis = np.array(fea_ou_dis)
            print(fea_ou_dis)
            result = fea_ou_dis[np.where((fea_ou_dis > 0.5))]
            print('sum:' + str(len(fea_ou_dis)))
            print(len(result))
            temp = np.random.randint(0, len(fea_ou_dis), 10)
            for dis_num in tqdm(range(10)):
                temp_type = str(class_num) + '_' + str(fea_num)
                df = df.append({'Feature': temp_type, 'Distance': fea_ou_dis[temp[dis_num]], 'Type': 'external'}, ignore_index=True)
    plt.figure(figsize=(60, 10))
    fig = sns.violinplot(x=df['Feature'], y=df['Distance'], hue=df['Type'])
    violinplot_fig = fig.get_figure()
    violinplot_fig.savefig(os.path.join(load_path, 'vgg16_violinplot.png'), dpi=400)
    plt.show()

def make_class_dis(load_path):
    df = pd.DataFrame(columns=('Category', 'Distance', 'Type'))  # 生成空的pandas表
    for class_num in range(10):
        print('class_num: ' + str(class_num))
        class_in_dis = np.load(os.path.join(load_path, 'in_dis', str(class_num) + '_in_dis.npy'), allow_pickle=True)
        class_ou_dis = np.load(os.path.join(load_path, 'ou_dis', str(class_num) + '_ou_dis.npy'), allow_pickle=True)
        for fea_num in range(len(class_in_dis)):
            print('fea_num: ' + str(fea_num))
            fea_in_dis = class_in_dis[fea_num]
            fea_in_dis = np.array(fea_in_dis)
            print(fea_in_dis)
            result = fea_in_dis[np.where((fea_in_dis < 0.5))]
            print('sum:' + str(len(fea_in_dis)))
            print(len(result))
            temp = np.random.randint(0, len(fea_in_dis), 10)
            for dis_num in tqdm(range(10)):
                # temp_type = str(class_num) + '_' + str(fea_num)
                df = df.append({'Category': class_num, 'Distance': fea_in_dis[temp[dis_num]], 'Type': 'inner'}, ignore_index=True)
        for fea_num in range(len(class_ou_dis)):
            print('fea_num: ' + str(fea_num))
            fea_ou_dis = class_ou_dis[fea_num]
            fea_ou_dis = np.array(fea_ou_dis)
            print(fea_ou_dis)
            result = fea_ou_dis[np.where((fea_ou_dis > 0.5))]
            print('sum:' + str(len(fea_ou_dis)))
            print(len(result))
            temp = np.random.randint(0, len(fea_ou_dis), 10)
            for dis_num in tqdm(range(10)):
                # temp_type = str(class_num) + '_' + str(fea_num)
                df = df.append({'Category': class_num, 'Distance': fea_ou_dis[temp[dis_num]], 'Type': 'external'}, ignore_index=True)
    plt.figure(figsize=(40, 10))
    # plt.xlabel('Category', fontsize=20)
    # plt.ylabel('Distance', fontsize=20)
    sns.set(font_scale=2)
    fig = sns.violinplot(x=df['Category'], y=df['Distance'], hue=df['Type'])
    violinplot_fig = fig.get_figure()
    violinplot_fig.savefig(os.path.join(load_path, 'class_violinplot.png'), dpi=400)
    plt.show()

def make_model_dis(load_path, model_list, dataset_list, model_name):
    data_num = 10000
    # data_num = 10
    df = pd.DataFrame(columns=('Model', 'Distance', 'Type'))  # 生成空的pandas表
    for model_num in range(len(model_list)):
        print('model_name:' + model_name[model_num])
        class_path = os.path.join(load_path, dataset_list[model_num], model_list[model_num], 'experiment/experiment_1')
        for class_num in range(10):
            print('class_num: ' + str(class_num))
            class_in_dis = np.load(os.path.join(class_path, 'in_dis', str(class_num) + '_in_dis.npy'), allow_pickle=True)
            class_ou_dis = np.load(os.path.join(class_path, 'ou_dis', str(class_num) + '_ou_dis.npy'), allow_pickle=True)
            for fea_num in range(len(class_in_dis)):
                print('fea_num: ' + str(fea_num))
                fea_in_dis = class_in_dis[fea_num]
                fea_in_dis = np.array(fea_in_dis)
                # print(fea_in_dis)
                result = fea_in_dis[np.where((fea_in_dis < 0.5))]
                print('sum:' + str(len(fea_in_dis)))
                print(len(result))
                temp = np.random.randint(0, len(fea_in_dis), data_num)
                for dis_num in tqdm(range(data_num)):
                    df = df.append({'Model': model_name[model_num], 'Distance': fea_in_dis[temp[dis_num]], 'Type': 'IS-values'}, ignore_index=True)
            for fea_num in range(len(class_ou_dis)):
                print('fea_num: ' + str(fea_num))
                fea_ou_dis = class_ou_dis[fea_num]
                fea_ou_dis = np.array(fea_ou_dis)
                # print(fea_ou_dis)
                result = fea_ou_dis[np.where((fea_ou_dis > 0.5))]
                print('sum:' + str(len(fea_ou_dis)))
                print(len(result))
                temp = np.random.randint(0, len(fea_ou_dis), data_num)
                for dis_num in tqdm(range(data_num)):
                    df = df.append({'Model': model_name[model_num], 'Distance': fea_ou_dis[temp[dis_num]], 'Type': 'ED-values'}, ignore_index=True)
            break
    plt.figure(figsize=(40, 10))
    sns.set(font_scale=2)
    fig = sns.violinplot(x=df['Model'], y=df['Distance'], hue=df['Type'])
    violinplot_fig = fig.get_figure()
    violinplot_fig.savefig(os.path.join(load_path, 'model_violinplot1.png'), dpi=400)
    plt.show()

if __name__ == '__main__':
    pass
