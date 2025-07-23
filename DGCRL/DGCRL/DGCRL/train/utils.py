import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_directory(path: str, sub_path_list: list):
    for sub_path in sub_path_list:
        if not os.path.exists(path + sub_path):
            os.makedirs(path + sub_path, exist_ok=True)
            print('Path: {} create successfully!'.format(path + sub_path))
        else:
            print('Path: {} is already existence!'.format(path + sub_path))


def plot_learning_curve(episodes, records, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, color='b', linestyle='-')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)

    plt.show()
    plt.savefig(figure_file)


def scale_action(action, low, high):
    action = np.clip(action, -1, 1)
    weight = (high - low) / 2
    bias = (high + low) / 2
    action_ = action * weight + bias

    return action_

def process_episodes(episodes):
    X = np.array(
        [np.concatenate((episode.states[:-1], episode.actions[:]), axis=1) for episode in episodes]).reshape(-1,5)
    Y = np.array([episode.states[1:] for episode in episodes]).reshape(-1, 4)
    print(f"inputs shape{X.shape}, outputs shape{Y.shape}")
    tensor_set = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
    return tensor_set

def extract_ood_data(episodes, ood_index, ood_flag):
    X, Y = [], []
    for i, episode in enumerate(episodes):
        start_index = ood_index[i] if ood_flag != 0 else 0
        states = np.array(episode.states[start_index:])
        actions = np.array(episode.actions[start_index:])
        X_ = np.concatenate((states[:-1], actions[:]), axis=1)
        Y_ = states[1:]
        X.append(X_)
        Y.append(Y_)
    X = np.vstack(X)
    Y = np.vstack(Y)
    return TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))


def evaluate_performance(labels, scores, threshold):
        tp = fp = tn = fn = 1e-10  # Add a small epsilon to avoid division by zero
        predictions = [1 if score > threshold else 0 for score in scores]
        for pred, label in zip(predictions, labels):
            if pred == label == 1:
                tp += 1
            elif pred == label == 0:
                tn += 1
            elif pred == 0 and label == 1:
                fn += 1
            elif pred == 1 and label == 0:
                fp += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1_score


def find_best_threshold(scores, labels, threshold_range):
        eval_list = []
        for threshold in threshold_range:
            precision, recall, f1 = evaluate_performance(labels, scores, threshold)
            eval_list.append((threshold, precision, recall, f1))
        # Find the max F1 score
        max_f1 = max(eval_list, key=lambda x: x[3])
        return max_f1