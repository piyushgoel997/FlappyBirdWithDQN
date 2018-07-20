from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
import Game as Game
import cv2
import random
import numpy as np
import time

LEARNING_RATE = 0.001
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.0001
REPLAY_MEMORY_SIZE = 10000
MINIBATCH_SIZE = 32
GAMMA = 0.99

NUM_ACTIONS = 2


def buildModel():
    model = Sequential()

    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(2))

    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


def train():
    # experience replay
    # epsilon-greedy exploration

    d = []  # replay memory

    game = Game.Game()

    model = buildModel()

    x_t, r_0, game_over = game.game_step([1, 0])  # first action is to do nothing

    # pre-processing
    x_t = cv2.cvtColor(x_t, cv2.COLOR_RGB2GRAY)
    x_t = cv2.resize(x_t, (84, 84))

    state_t = np.stack((x_t, x_t, x_t, x_t), axis=2).astype(np.float32)
    state_t = state_t.reshape(1, *state_t.shape)

    # training
    iteration = 1
    epsilon = INITIAL_EPSILON
    while True:
        start_time = time.time()
        action_t = np.zeros([NUM_ACTIONS])

        if random.random() <= epsilon:
            # do random action
            action_t[random.randrange(NUM_ACTIONS)] = 1
        else:
            predicted_q_value = model.predict(state_t)
            max_q = np.argmax(predicted_q_value)
            action_t[max_q] = 1

        # carry out action a
        x_t_next, reward_t, game_over = game.game_step(action_t)

        x_t_next = cv2.cvtColor(x_t_next, cv2.COLOR_RGB2GRAY)
        x_t_next = cv2.resize(x_t_next, (84, 84))
        x_t_next = x_t_next.reshape(1, *x_t_next.shape, 1).astype(np.float32)


        # take the last 3 images from the prev state and add the current image to it, to get the new state.
        state_t_next = np.append(x_t_next, state_t[:, :, :, :3], axis=3)

        # store the transition (curr state, action taken, reward, next state, game over state) in the replay memory
        d.append((state_t, np.argmax(action_t), reward_t, state_t_next, game_over))

        if len(d) > REPLAY_MEMORY_SIZE:
            d.remove(d[0])

        # only do the training after 320 iterations are complete
        # experience replay
        loss = 0
        if iteration > 320:
            # sampling the minibatch for training
            minibatch = random.sample(d, MINIBATCH_SIZE)
            input = np.zeros((MINIBATCH_SIZE, *state_t.shape[1:])).astype(np.float32)
            target = np.zeros((MINIBATCH_SIZE, NUM_ACTIONS)).astype(np.float32)

            for i in range(MINIBATCH_SIZE):
                s = minibatch[i][0]
                a = minibatch[i][1]
                r = minibatch[i][2]
                s1 = minibatch[i][3]
                over = minibatch[i][4]

                input[i] = s
                target[i] = model.predict(s)
                q1 = model.predict(s1)

                if over:
                    target[i, a] = reward_t
                else:
                    target[i, a] = reward_t + GAMMA * np.max(q1)

            loss += model.train_on_batch(input, target)

        state_t = state_t_next
        iteration += 1

        if iteration % 1000 == 0:
            model.save_weights("model", overwrite=True)
            print("model saved")

        print("iteration " + str(iteration) +
              " epsilon " + str(epsilon) +
              " loss " + str(loss) +
              " in time " + str(time.time() - start_time))

        # reduce the epsilon gradually
        if iteration > 320 and epsilon > FINAL_EPSILON:
            epsilon -= 1.0e-6


if __name__ == '__main__':
    train()
