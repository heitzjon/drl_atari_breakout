import gym
import random
import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Model

from collections import deque
from keras.optimizers import RMSprop
from keras import backend as K
from datetime import datetime
import os.path
import time
from keras.models import load_model
from keras.models import clone_model


# CONSTANTS
GAME = 'BreakoutDeterministic-v4'

PATH = 'train_breakout'
FILE = 'breakout_model'
ABS_PATH_TESTING = PATH+'/'+FILE+'19993112235959.h5'

ATARI_SHAPE = (105, 80, 4)
ACTION_SIZE = 3

BATCH_SIZE = 32

REFRESH_TARGET_MODEL = 10000

REPLAY_MEMORY_SIZE = 250000 #takes up to 13 GB to store this amount of history data

OBSERVATION_STEPS = 50000
EXPLORATION_STEPS = 1000000

EPISODES_NUM = 100000

EPSILON_INIT = 1.0
EPSILON_FINAL = 0.01

GAMMA = 0.99




# FRAME PREPROCESSING
def to_grayscale(img):
  return np.mean(img, axis=2).astype(np.uint8)

def downsample(img):
  return img[::2, ::2]

def pre_processing(img):
  return to_grayscale(downsample(img))


# HUBER LOSS (LIKE MSE JUST WITH A LINEAR PART ON THE BOTH ENDS)
def huber_loss(y, q_value):
    error = K.abs(y - q_value)
    quadratic_part = K.clip(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
    return loss


# NN MODEL
def atari_model():
    # define inputs.
    frames_input = layers.Input(ATARI_SHAPE, name='frames')
    actions_input = layers.Input((ACTION_SIZE,), name='action_mask')

    # normalization layer ([0,255] transforming to [0, 1])
    normalized = layers.Lambda(lambda x: x / 255.0, name='normalization')(frames_input)

    # define hidden layers
    conv_1 = layers.convolutional.Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(normalized)
    conv_2 = layers.convolutional.Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv_1)
    conv_flattened = layers.core.Flatten()(conv_2)
    hidden = layers.Dense(256, activation='relu')(conv_flattened) #fully connected

    # define output layer (fully connected, linear activation)
    output = layers.Dense(ACTION_SIZE)(hidden)

    # multiplication to update qvalues
    filtered_output = layers.Multiply(name='QValue')([output, actions_input])

    # create and compile model
    model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    model.summary()
    optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)

    model.compile(optimizer, loss=huber_loss)
    return model


# GET BEST ACTION (epsilon-greedy policy, in 3 phases(observation,exploration with annealing,fixed epsilon))
def get_action(history, epsilon, step, model):
    if np.random.rand() <= epsilon or step <= OBSERVATION_STEPS:
        return random.randrange(ACTION_SIZE)
    else:
        q_value = model.predict([history, np.ones(ACTION_SIZE).reshape(1, ACTION_SIZE)])
        return np.argmax(q_value[0])


# MEMORICE SAMPLE <s,a,r,s'> FOR REPLAY
def store_memory(memory, state, action, reward, next_state, dead):
    memory.append((state, action, reward, next_state, dead))

# ONE HOT ENCODE => FOR ACTIONS
def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


#FIT BATCH
def fit_batch(memory, model):
    mini_batch = random.sample(memory, BATCH_SIZE)
    state = np.zeros((BATCH_SIZE, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
    next_state = np.zeros((BATCH_SIZE, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
    target = np.zeros((BATCH_SIZE,))
    action, reward, dead = [], [], []

    for idx, val in enumerate(mini_batch):
        state[idx] = val[0]
        action.append(val[1])
        reward.append(val[2])
        next_state[idx] = val[3]
        dead.append(val[4])

    actions_mask = np.ones((BATCH_SIZE, ACTION_SIZE))
    next_Q_values = model.predict([next_state, actions_mask])

    # like Q Learning, get maximum Q value at s'
    # But from target model
    for i in range(BATCH_SIZE):
        if dead[i]:
            target[i] = -1 # target[i] = reward[i]
        else:
            target[i] = reward[i] + GAMMA * np.amax(next_Q_values[i])

    action_one_hot = get_one_hot(action, ACTION_SIZE)
    target_one_hot = action_one_hot * target[:, None]


    h = model.fit([state, action_one_hot], target_one_hot, epochs=1, batch_size=BATCH_SIZE, verbose=0)

    return h.history['loss'][0]

# TRAIN
def train():
    env = gym.make(GAME)

    memory = deque(maxlen=REPLAY_MEMORY_SIZE)
    episode_number = 0
    epsilon = EPSILON_INIT
    epsilon_decay = (EPSILON_INIT - EPSILON_FINAL) / EXPLORATION_STEPS
    global_step = 0

    model = atari_model()

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = "{}/run-{}-log".format(PATH, now)
    file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

    #clone model
    model_target = clone_model(model)
    model_target.set_weights(model.get_weights())

    while episode_number < EPISODES_NUM:

        done = False
        dead = False
        # 1 episode = 5 lives
        step, score, start_life = 0, 0, 5
        loss = 0.0
        observe = env.reset()


        # do nothing at the start of episode to avoid sub-optimal
        for _ in range(random.randint(1, 30)):
            observe, _, _, _ = env.step(1)
        # at start of episode, there is no preceding frame
        # So just copy initial states to make history
        frame = pre_processing(observe)
        state = np.stack((frame, frame, frame, frame), axis=2)
        state = np.reshape([state], (1,105,80,4))

        while not done:

            # get action for the current history and go one step in environment
            action = get_action(state, epsilon, global_step, model_target)
            real_action = action + 1

            # scale down epsilon, but only after observe steps
            if epsilon > EPSILON_FINAL and global_step > OBSERVATION_STEPS:
                epsilon -= epsilon_decay

            observe, reward, done, info = env.step(real_action)
            frame = pre_processing(observe)
            frame = np.reshape([frame], (1,105,80,1))
            next_state = np.append(frame, state[:, :, :, :3], axis=3)

            # check if agent is dead
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            # store to memory
            store_memory(memory, state, action, reward, next_state, dead)  #

            # check if the memory is ready for training
            if global_step > OBSERVATION_STEPS:
                loss = loss + fit_batch(memory, model)
                if global_step % REFRESH_TARGET_MODEL == 0:  # update the target model
                    model_target.set_weights(model.get_weights())

            score += reward

            if dead:
                dead = False
            else:
                state = next_state

            global_step += 1
            step += 1

            # write logs and weights to disk as soon as episode is over
            if done:
                if global_step <= OBSERVATION_STEPS:
                    state = "observe"
                elif OBSERVATION_STEPS < global_step <= OBSERVATION_STEPS + EXPLORATION_STEPS:
                    state = "explore"
                else:
                    state = "train"
                print('state: {}, episode: {}, score: {}, global_step: {}, avg loss: {}, step: {}, memory length: {}'
                      .format(state, episode_number, score, global_step, loss / float(step), step, len(memory)))

                if episode_number % 100 == 0 or (episode_number + 1) == EPISODES_NUM:
                    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                    file_name = FILE+"_{}.h5".format(now)
                    model_path = os.path.join(PATH, file_name)
                    model.save(model_path)

                # Add user custom data to TensorBoard
                loss_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="loss", simple_value=loss / float(step))])
                file_writer.add_summary(loss_summary, global_step=episode_number)

                score_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="score", simple_value=score)])
                file_writer.add_summary(score_summary, global_step=episode_number)

                episode_number += 1

    file_writer.close()


def test():
    env = gym.make(GAME)

    episode_number = 0
    epsilon = 0.001
    global_step = OBSERVATION_STEPS+1

    model = load_model(ABS_PATH_TESTING, custom_objects={'huber_loss': huber_loss})


    while episode_number < 10:

        done = False
        dead = False
        # 1 episode = 5 lives
        score, start_life = 0, 5
        env.reset()

        observe, _, _, _ = env.step(1)
        # At start of episode, there is no preceding frame
        # So just copy initial states to make history
        frame = pre_processing(observe)
        state = np.stack((frame, frame, frame, frame), axis=2)
        state = np.reshape([state], (1,105,80,4))

        while not done:
            env.render()
            time.sleep(0.05)

            # get action for the current history and go one step in environment
            action = get_action(state, epsilon, global_step, model)
            # change action to real_action
            real_action = action + 1

            observe, reward, done, info = env.step(real_action)
            # pre-process the observation
            frame = pre_processing(observe)
            frame = np.reshape([frame], (1,105,80,1))
            next_state = np.append(frame, state[:, :, :, :3], axis=3)

            # if the agent missed ball, agent is dead --> episode is not over
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']


            score += reward

            # If agent is dead, set the flag back to false, but keep the history unchanged,
            # to avoid to see the ball up in the sky
            if dead:
                dead = False
            else:
                state = next_state

            global_step += 1

            if done:
                episode_number += 1
                print('episode: {}, score: {}'.format(episode_number, score))



#train()
test()
