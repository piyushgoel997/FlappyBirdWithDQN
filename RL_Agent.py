import argparse
import tensorflow as tf
import Game as Game
import cv2
import random
import numpy as np
import time
from Model import model

LEARNING_RATE = 0.001
INITIAL_EPSILON = 0.2
FINAL_EPSILON = 0.0001
REPLAY_MEMORY_SIZE = 10000
MINIBATCH_SIZE = 64
GAMMA = 0.99

NUM_ACTIONS = 2


def run(args):
    # experience replay
    # epsilon-greedy exploration

    log_file = open('training.log', 'a')

    X = tf.placeholder(dtype=tf.float32, shape=(None, 84, 84, 4), name='X')
    Y = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='Y')

    pred = model(X)

    with tf.name_scope('train'):
        loss = tf.reduce_mean(tf.pow(Y - pred, 2))
        tf.summary.scalar('loss', loss)
        train_op = tf.train.AdamOptimizer().minimize(loss)

    summary = tf.summary.merge_all()

    d = []  # replay memory

    game = Game.Game()

    x_t, r_0, game_over = game.game_step([1, 0])  # first action is to do nothing

    # pre-processing
    x_t = cv2.cvtColor(x_t, cv2.COLOR_RGB2GRAY)
    x_t = cv2.resize(x_t, (84, 84))

    state_t = np.stack((x_t, x_t, x_t, x_t), axis=2).astype(np.float32)
    state_t = state_t.reshape(1, *state_t.shape)

    # training
    iteration = 1
    epsilon = INITIAL_EPSILON

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        if args.reload:
            restore_path = tf.train.latest_checkpoint("model")
            saver.restore(sess, restore_path)
            print('restored the model from' + str(restore_path))
            log_file.write('\n\nrestored the model from' + str(restore_path))
        else:
            sess.run(init)

        if args.test:
            total_rew = r_0
            while not game_over:
                # predict action
                action_t = np.zeros([NUM_ACTIONS])
                pred_q = sess.run(pred, feed_dict={X: state_t})
                action_t[np.argmax(pred_q)] = 1

                # execute the action
                x_t_next, r_t, game_over = game.game_step(action_t)
                x_t_next = cv2.cvtColor(x_t_next, cv2.COLOR_RGB2GRAY)
                x_t_next = cv2.resize(x_t_next, (84, 84))
                x_t_next = x_t_next.reshape(1, *x_t_next.shape, 1).astype(np.float32)

                state_t = np.append(x_t_next, state_t[:, :, :, :3], axis=3)

                total_rew += r_t

            print(total_rew)
            return

        summary_writer = tf.summary.FileWriter('summary', sess.graph)

        while iteration <= args.iterations:
            start_time = time.time()

            action_t = np.zeros([NUM_ACTIONS])

            if random.random() <= epsilon:
                # do random action
                action_t[random.randrange(NUM_ACTIONS)] = 1
            else:
                predicted_q_value = sess.run(pred, {X: state_t})
                max_q = np.argmax(predicted_q_value)
                action_t[max_q] = 1

            # carry out action a
            x_t_next, reward_t, game_over = game.game_step(action_t)

            x_t_next = cv2.cvtColor(x_t_next, cv2.COLOR_RGB2GRAY)
            x_t_next = cv2.resize(x_t_next, (84, 84))
            x_t_next = x_t_next.reshape(1, *x_t_next.shape, 1).astype(np.float32)

            state_t = np.append(x_t_next, state_t[:, :, :, :3], axis=3)

            # take the last 3 images from the prev state and add the current image to it, to get the new state.
            state_t_next = np.append(x_t_next, state_t[:, :, :, :3], axis=3)

            # store the transition (curr state, action taken, reward, next state, game over state) in the replay memory
            d.append((state_t, np.argmax(action_t), reward_t, state_t_next, game_over))

            if len(d) > REPLAY_MEMORY_SIZE:
                d.remove(d[0])

            # only do the training after 320 iterations are complete
            # experience replay
            iter_loss = 0
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
                    target[i] = sess.run(pred, feed_dict={X: s})
                    q1 = sess.run(pred, feed_dict={X: s1})

                    if over:
                        target[i, a] = reward_t
                    else:
                        target[i, a] = reward_t + GAMMA * np.max(q1)

                _loss, _, _summary = sess.run([loss, train_op, summary], feed_dict={X: input, Y: target})
                summary_writer.add_summary(_summary)
                iter_loss += _loss

            state_t = state_t_next
            iteration += 1

            if iteration % 1000 == 0:
                save_path = saver.save(sess, 'model/model.ckpt')
                print("Model saved in the dir " + str(save_path))
                log_file.write("\nModel saved in the dir " + str(save_path))

            print("iteration " + str(iteration) +
                  " epsilon " + str(epsilon) +
                  " loss " + str(loss) +
                  " in time " + str(time.time() - start_time))
            log_file.write("\niteration " + str(iteration) +
                           " epsilon " + str(epsilon) +
                           " loss " + str(loss) +
                           " in time " + str(time.time() - start_time))

            # reduce the epsilon gradually
            if iteration > 320 and epsilon > FINAL_EPSILON:
                epsilon -= 1.0e-6


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--reload', type=bool, default=False)
    parser.add_argument('--iterations', type=int, default=100000)

    args = parser.parse_args()

    run(args)
