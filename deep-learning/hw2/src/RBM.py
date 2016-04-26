# coding=utf-8

import sys

import numpy as np


get_x = lambda t: np.array(zip(*t)[0], dtype=np.float32)

sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
hamming = lambda y, z: np.sum(np.abs(y - z), axis=1)
sample = lambda p: (p > np.random.uniform(0, 1, p.shape)).astype(np.float32)


class RBM:
    def __init__(self, size, eta, batch_size, epochs):
        self.v_size, self.h_size = size
        self.eta = eta
        self.batch_size = batch_size
        self.epochs = epochs

        self.w = np.random.normal(0, 0.1, (self.v_size, self.h_size))
        self.a = np.zeros(self.v_size)
        self.b = np.zeros(self.h_size)

    def hidden_step(self, visible):
        return sigmoid(np.dot(visible, self.w) + self.b)

    def visible_step(self, hidden):
        return sigmoid(np.dot(hidden, self.w.T) + self.a)

    def contrastive_divergence(self, batch):
        vis = batch
        nabla_w = np.zeros(self.w.shape)
        nabla_a = np.zeros(self.a.shape)
        nabla_b = np.zeros(self.b.shape)

        # CD-k, уже при k = 1 качество не сильно уступает большим значениям,
        # но выигрыш в скорости значительный => будем делать только один проход без цикла.

        p_hid = self.hidden_step(vis)
        nabla_w += np.dot(vis.T, p_hid)
        nabla_a += np.sum(vis, axis=0)
        nabla_b += np.sum(p_hid, axis=0)

        hid = sample(p_hid)

        p_vis = self.visible_step(hid)
        # vis = sample(p_vis)
        # не семплировать видимый слой
        # (семплирование замедляет сходимость, но математически это более корректно);
        vis = p_vis

        p_hid = self.hidden_step(vis)

        # не семплировать значения скрытого слоя при выводе из восстановленного образа;
        nabla_w -= np.dot(vis.T, p_hid)
        nabla_a -= np.sum(vis, axis=0)
        nabla_b -= np.sum(p_hid, axis=0)

        nabla_w /= np.float32(self.batch_size)
        nabla_a /= np.float32(self.batch_size)
        nabla_b /= np.float32(self.batch_size)

        self.w += self.eta * nabla_w
        self.a += self.eta * nabla_a
        self.b += self.eta * nabla_b

        #         batch = sample(p_vis)

        energy = (-np.sum(self.a * batch) - np.sum(self.b * hid) - np.sum(np.dot(batch.T, hid) * self.w))
        energy /= batch.shape[0]

        return energy

    def fit(self, train_data, cv_data=None):
        scores = []
        energies = []

        for epoch in range(self.epochs):
            np.random.shuffle(train_data)  # inplace shuffle
            energy = self.contrastive_divergence(train_data[:self.batch_size])
            energies.append(energy)

            if cv_data is not None:
                pred_data = sample(self.visible_step(sample(self.hidden_step(cv_data))))
                score = np.mean(hamming(cv_data, pred_data))

                scores.append(score)

                sys.stdout.write('\r' + "%s %s" % (score, energy))
                sys.stdout.flush()

        return scores, energies
