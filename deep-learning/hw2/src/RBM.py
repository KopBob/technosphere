# coding=utf-8

import sys

import numpy as np

get_x = lambda t: np.array(zip(*t)[0], dtype=np.float32)

sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
hamming = lambda y, z: np.sum(np.abs(y - z), axis=1)
sample = lambda p: (p > np.random.uniform(0, 1, p.shape)).astype(np.float32)


class RBM:
    def __init__(self, size, eta, batch_size, epochs,
                 mode='bern', momentum=0.0, w=None, a=None, b=None):
        self.v_size, self.h_size = size
        self.eta = eta
        self.batch_size = batch_size
        self.epochs = epochs
        self.mode = mode
        self.momentum = momentum

        self.w = np.random.normal(0, 0.1, (self.v_size, self.h_size)) if w is None else w
        self.a = np.zeros(self.v_size) if a is None else a
        self.b = np.zeros(self.h_size) if b is None else b

        self.delta_w = np.zeros((self.v_size, self.h_size))
        self.delta_a = np.zeros(self.v_size)
        self.delta_b = np.zeros(self.h_size)

        self.scores = []

    def hidden_step(self, visible):
        return sigmoid(np.dot(visible, self.w) + self.b)

    def visible_step(self, hidden):
        if self.mode == 'bern':
            return sigmoid(np.dot(hidden, self.w.T) + self.a)
        elif self.mode == 'gauss':
            return np.dot(hidden, self.w.T) + self.a

    def contrastive_divergence(self, batch):
        self.delta_w *= self.momentum
        self.delta_a *= self.momentum
        self.delta_b *= self.momentum

        vis = batch

        # CD-k, уже при k = 1 качество не сильно уступает большим значениям,
        # но выигрыш в скорости значительный => будем делать только один проход без цикла.
        # P(h|v)
        p_hid = self.hidden_step(vis)

        self.delta_w += np.dot(vis.T, p_hid)
        self.delta_a += np.sum(vis, axis=0)
        self.delta_b += np.sum(p_hid, axis=0)

        hid = 1. * sample(p_hid)

        p_vis = self.visible_step(hid)

        # не семплировать видимый слой
        # (семплирование замедляет сходимость, но математически это более корректно);
        p_hid = self.hidden_step(p_vis)

        # не семплировать значения скрытого слоя при выводе из восстановленного образа;
        self.delta_w -= np.dot(p_vis.T, p_hid)
        self.delta_a -= np.sum(p_vis, axis=0)
        self.delta_b -= np.sum(p_hid, axis=0)

        self.w += self.eta * self.delta_w / self.batch_size
        self.a += self.eta * self.delta_a / self.batch_size
        self.b += self.eta * self.delta_b / self.batch_size

    def fit(self, train_data, cv_data=None):

        for epoch in range(self.epochs):
            np.random.shuffle(train_data)  # inplace shuffle

            for i in range(len(train_data) / self.batch_size):
                batch = train_data[self.batch_size * i:self.batch_size * i + self.batch_size]
                self.contrastive_divergence(batch)

                if cv_data is not None:
                    pred_data = self.visible_step(sample(self.hidden_step(cv_data)))
                    score = np.mean(((cv_data - pred_data) ** 2))
                    self.scores.append(score)

                    sys.stdout.write('\r' + "%s / %s | %s" \
                                     % (epoch, self.epochs, self.scores[-1]))
                    sys.stdout.flush()
            #
            # if cv_data is not None:
            #     pred_data = self.visible_step(sample(self.hidden_step(cv_data)))
            #     score = np.mean(np.sum((cv_data - pred_data) ** 2, axis=1))
            #     self.scores.append(score)
            #
            #     score_delta = 0
            #     if len(self.scores) > 10:
            #         score_delta = np.abs(np.mean(np.diff(self.scores[-20:])))
            #
            #         if score_delta <= 0.001:
            #             print("!", score_delta)
            #             break
            #     sys.stdout.write('\r' + "%s / %s | %s %s" \
            #                      % (epoch, self.epochs, self.scores[-1], score_delta))
            #     sys.stdout.flush()


