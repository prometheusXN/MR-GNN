# based on para9
# concat to matmul
# concat in the end of matmul, different from para17

import deepchem as dc
from deepchem.models.tf_new_models.graph_topology import GraphTopology
import os
import tensorflow as tf
import numpy as np
# import models.layers as mylayer
import argparse
import layers_new_3 as mylayers
from data_load_2 import load_interaction_data
from deepchem.models.tf_new_models.graph_topology import merge_dicts
import sys
import time,pickle
from sklearn import metrics

model_name='para_test9_matmul_coLSTM'
conv_size=384
Dense_size=128
hidden_size=384
tensor_size=200

n_bins=25

dropout0,dropout1,dropout2,dropout3,dropout4,dropout5=None, None, None, 0., 0., 0.
# dropout0,dropout1,dropout2,dropout3,dropout4,dropout5=[0.]*6
learning_rate=0.0001
beta=0.001
batch_size = 512
dataset='c900'
# dataset='c900'
merge_mode= 'concat'
pairwise_node_comparison=False
reverse=True
# mode='prediction_vector'
mode='training'

print(model_name)
FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument('--optimize_type', type=str, default='adam', help='Optimizer type.')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate.')
parser.add_argument('--lambda_l2', type=float, default=1e-5, help='The coefficient of L2 regularizer.')
parser.add_argument('--max_epochs', type=int, default=600, help='Maximum epochs for training.')
parser.add_argument('--frac_train', type=float, default=.72, help='frac_train')
parser.add_argument('--S_LSTM', type=bool, default=True, help='S_LSTM')
parser.add_argument('--I_LSTM', type=bool, default=True, help='I_LSTM')

sys.stdout.flush()
FLAGS, unparsed = parser.parse_known_args()

# learning_rate = FLAGS.learning_rate
optimize_type = FLAGS.optimize_type
lambda_l2 = FLAGS.lambda_l2
max_epochs = FLAGS.max_epochs
frac_train=FLAGS.frac_train
S_LSTM=FLAGS.S_LSTM
I_LSTM=FLAGS.I_LSTM

filename='test.{}-conv{}-dense{}-t_size{}-hid{}-drop2{}-learn{}-beta{}-data-{}-batch{}-{}-r{}-{}.{}-{}'.format\
    (model_name, conv_size, Dense_size, tensor_size, hidden_size, dropout2, learning_rate, beta, dataset, batch_size, merge_mode,reverse,
     time.localtime(time.time()).tm_mon, time.localtime(time.time()).tm_mday, time.localtime(time.time()).tm_hour)
current_dir = os.path.dirname(os.path.realpath(__file__))
best_path =current_dir +  '/model_save/' + filename
# restore_path = current_dir +  '/model_save/'+'test.model256-256-256-0.0-0.0001-0.001-c900-b512-para_test9-1'
restore_path = None
print('restore_path:',restore_path)
print(best_path)
n_features = 75

# dropout = 0.5
# h_size = 256
n_class = 2
best_accuracy = 0.0

has_pre_trained_model = False
if (restore_path is not None and os.path.exists(restore_path + '.meta')):
    has_pre_trained_model = True
    print('model exist in', restore_path)
if os.path.exists(best_path + '.meta'):
    has_pre_trained_model = True
    restore_path=best_path
    print('model exist in', restore_path)


def evaluate(dataset_a, dataset_b, steps, lose_num):
    total_tags = 0.0
    correct_tags = 0.0
    predic,auc_y_set, auc_pred_set, loss_sum = [],[], [], 0

    data_a = dataset_a.iterbatches(batch_size, epoch=max_epochs, pad_batches=True, deterministic=True)
    data_b = dataset_b.iterbatches(batch_size, epoch=max_epochs, pad_batches=True, deterministic=True)

    def scores(tp, fp, tn, fn):
        print((tp, fp, tn, fn), tp + tn + fp + fn)
        acc = (tp + tn) / (tp + tn + fp + fn)
        tpr = tp / (tp + fn)
        tnn = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
        return (acc, tpr, tnn, ppv, npv, f1)

    for i in range(steps):
        num_y = batch_size
        if i + 1 == steps:
            num_y = batch_size - lose_num

        a,b=next(data_a), next(data_b)

        total_tags += num_y
        correct_tag, predic_, y_, pred_ = sess.run(
            (correct, predictions, label_gold, pred), feed_dict=gen_dict(a, b))
        predic+=list(predic_[:num_y])
        auc_y_set += list(y_[:num_y])
        auc_pred_set += list(pred_[:num_y])
        correct_tag = correct_tag[:num_y]
        correct_tags += np.sum(np.cast[np.int32](correct_tag))

    fpr, tpr, thresholds = metrics.roc_curve(auc_y_set, auc_pred_set, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    accuracy_1 = correct_tags / total_tags * 100
    tp = np.sum(np.float32(np.logical_and(np.equal(predic,1), np.equal(auc_y_set, 1))))
    fp = np.sum(np.float32(np.logical_and(np.equal(predic,1), np.equal(auc_y_set, 0))))
    tn = np.sum(np.float32(np.logical_and(np.equal(predic,0), np.equal(auc_y_set, 0))))
    fn = np.sum(np.float32(np.logical_and(np.equal(predic,0), np.equal(auc_y_set, 1))))
    accuracy = scores(tp, fp, tn, fn)
    # print(auc_y_set)
    return accuracy_1, auc, accuracy


with tf.Graph().as_default():
    graphA_topology = GraphTopology(n_features, name='topology_A')
    graphB_topology = GraphTopology(n_features, name='topology_B')
    with tf.name_scope('input'):
        outputA = graphA_topology.get_atom_features_placeholder()
        outputB = graphB_topology.get_atom_features_placeholder()
        label_gold = tf.placeholder(dtype=tf.int32, shape=[None], name="label_placeholder")
        training = tf.placeholder(dtype='float32', shape=(), name='ops_training')
    add_time = 0

    lstm = tf.contrib.rnn.BasicLSTMCell(Dense_size)
    hidden_state = tf.zeros([batch_size, lstm.state_size[0]])
    current_state = tf.zeros([batch_size, lstm.state_size[0]])
    
    lstm_1 = tf.contrib.rnn.BasicLSTMCell(Dense_size*2)
    Hidden_state = tf.zeros([batch_size, lstm_1.state_size[0]])
    Current_state = tf.zeros([batch_size, lstm_1.state_size[0]])
    
    stateA = hidden_state, current_state
    stateB = hidden_state, current_state
    
    State = Hidden_state, Current_state

    # layer1 :
    lay1_conv = mylayers.GraphConv_and_gather(conv_size, n_features, batch_size, activation='relu', dropout=dropout0)
    # lay1_norm = dc.nn.BatchNormalization(epsilon=1e-5, mode=1)
    lay1_pool = dc.nn.GraphPool()
    lay1_dense = mylayers.Dense(Dense_size, conv_size, activation='relu')

    outputA, gatherA, _ = lay1_conv(
        [outputA] + graphA_topology.get_topology_placeholders() + [training] + [add_time])
    # outputA = lay1_norm(outputA)
    outputA = lay1_pool([outputA] + graphA_topology.get_topology_placeholders() + [training] + [add_time])
    gatherA = lay1_dense(gatherA)

    # gatherA = lay1_norm(gatherA)
    h_A_1, stateA = lstm(gatherA, stateA)

    outputB, gatherB, _ = lay1_conv(
        [outputB] + graphB_topology.get_topology_placeholders() + [training] + [add_time])
    # outputB = lay1_norm(outputB)
    outputB = lay1_pool([outputB] + graphB_topology.get_topology_placeholders() + [training] + [add_time])
    gatherB = lay1_dense(gatherB)

    # gatherB = lay1_norm(gatherB)
    h_B_1, stateB = lstm(gatherB, stateB)

    Inter = tf.concat([gatherA, gatherB], 1)
    h_1, State = lstm_1(Inter, State)

    # layer2 :
    lay2_conv = mylayers.GraphConv_and_gather(conv_size, conv_size, batch_size, activation='relu', dropout=dropout1)
    # lay2_norm = dc.nn.BatchNormalization(epsilon=1e-5, mode=1)
    lay2_pool = dc.nn.GraphPool()
    lay2_dense = mylayers.Dense(Dense_size, conv_size, activation='relu')

    outputA, gatherA, _ = lay2_conv(
        [outputA] + graphA_topology.get_topology_placeholders() + [training] + [add_time])
    # outputA = lay2_norm(outputA)
    outputA = lay2_pool([outputA] + graphA_topology.get_topology_placeholders() + [training] + [add_time])
    gatherA = lay2_dense(gatherA)

    # gatherA = lay2_norm(gatherA)
    h_A_2, stateA = lstm(gatherA, stateA)

    outputB, gatherB, _ = lay2_conv(
        [outputB] + graphB_topology.get_topology_placeholders() + [training] + [add_time])
    # outputB = lay2_norm(outputB)
    outputB = lay2_pool([outputB] + graphB_topology.get_topology_placeholders() + [training] + [add_time])
    gatherB = lay2_dense(gatherB)

    # gatherB = lay2_norm(gatherB)
    h_B_2, stateB = lstm(gatherB, stateB)
    
    Inter = tf.concat([gatherA, gatherB], 1)
    h_2, State = lstm_1(Inter, State)

    # # layer3 :
    # lay3_conv = mylayers.GraphConv_and_gather(conv_size, conv_size, batch_size, activation='relu', dropout=dropout)
    # lay3_norm = dc.nn.BatchNormalization(epsilon=1e-5, mode=1)
    # lay3_pool = dc.nn.GraphPool()
    # lay3_dense = mylayers.Dense(Dense_size, conv_size, activation='relu')
    #
    # outputA, gatherA = lay3_conv([outputA] + graphA_topology.get_topology_placeholders() + [training] + [add_time])
    # outputA = lay3_norm(outputA)
    # outputA = lay3_pool([outputA] + graphA_topology.get_topology_placeholders() + [training] + [add_time])
    # gatherA = lay3_dense(gatherA)
    # gatherA = lay3_norm(gatherA)
    # h_A_3, stateA = lstm(gatherA, stateA)
    #
    # outputB, gatherB = lay3_conv([outputB] + graphB_topology.get_topology_placeholders() + [training] + [add_time])
    # outputB = lay3_norm(outputB)
    # outputB = lay3_pool([outputB] + graphB_topology.get_topology_placeholders() + [training] + [add_time])
    # gatherB = lay3_dense(gatherB)
    # gatherB = lay3_norm(gatherB)
    # h_B_3, stateB = lstm(gatherB, stateB)

    # layer4 :
    lay4_gather = mylayers.GraphConv_and_gather(conv_size, conv_size, batch_size, activation='relu', dropout=dropout2)
    # lay4_norm = dc.nn.BatchNormalization(epsilon=1e-5, mode=1)
    lay4_dense = mylayers.Dense(Dense_size, conv_size, activation='relu')

    outputA, gatherA, poolingA = lay4_gather(
        [outputA] + graphA_topology.get_topology_placeholders() + [training] + [add_time])
    gatherA = lay4_dense(gatherA)

    # gatherA = lay4_norm(gatherA)
    h_A_4, stateA = lstm(gatherA, stateA)
    

    outputB, gatherB, poolingB = lay4_gather(
        [outputB] + graphB_topology.get_topology_placeholders() + [training] + [add_time])
    gatherB = lay4_dense(gatherB)

    # gatherB = lay4_norm(gatherB)
    h_B_4, stateB = lstm(gatherB, stateB)
    
    Inter = tf.concat([gatherA, gatherB], 1)
    h_4, State = lstm_1(Inter, State)
    # mul=[]
    # for i in range(batch_size):
    #     a=tf.slice(h_A_4,[i,0],[1,-1])
    #     a=tf.reshape(a,[-1,1])
    #     b = tf.slice(h_B_4, [i, 0], [1, -1])
    #     mul.append(tf.reshape(tf.matmul(a, b),[-1]))
    # mul=tf.convert_to_tensor(mul)
    fully_size = 0
    if S_LSTM:
        h_A_4 = tf.concat([h_A_4, poolingA], 1)
        h_B_4 = tf.concat([h_B_4, poolingB], 1)
        fully_size += Dense_size * 2 + conv_size * 2
    else:
        h_A_4 = gatherA
        h_B_4 = gatherB
        fully_size += Dense_size * 2

    if merge_mode == 'concat':
        # fully_size = Dense_size* 4
        if I_LSTM:
            fully_size += Dense_size * 2
            A_B = tf.concat([h_A_4, h_B_4, h_4], 1)
        else:
            A_B = tf.concat([h_A_4, h_B_4], 1)
    elif merge_mode == 'add':
        fully_size = Dense_size + conv_size
        A_B = h_A_4 + h_B_4
    elif merge_mode == 'add&':
        fully_size = (Dense_size + conv_size) * 2
        A_B = tf.concat([h_A_4 + h_B_4, tf.abs(h_A_4 - h_B_4)], -1)
    else:
        with tf.name_scope('neural_tensor_network'):
            # Tensor_size = 2 * Dense_size
            # interaction_size = 4 * Dense_size
            Tensor_size = Dense_size + conv_size
            interaction_size = 2 * Tensor_size
            with tf.name_scope('W_tensor'):
                w_tensor = tf.get_variable('w_tensor', shape=[tensor_size, Tensor_size, Tensor_size],
                                           dtype=tf.float32,
                                           initializer=tf.contrib.layers.xavier_initializer())
                # variable_summaries(w_tensor)
            with tf.name_scope('B_tensor'):
                b_tensor = tf.get_variable("b_tensor", [tensor_size], dtype=tf.float32)
                # variable_summaries(b_tensor)
            with tf.name_scope('W_interaction'):
                W_interaction = tf.get_variable('W_interaction', shape=[interaction_size, tensor_size],
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())
                # variable_summaries(W_interaction)
            A_B = tf.map_fn(lambda W: h_A_4 @ W, w_tensor, dtype=tf.float32)
            A_B = tf.transpose(A_B, [1, 0, 2])
            A_B = A_B @ tf.expand_dims(h_B_4, -1)
            A_B = tf.reduce_sum(A_B, axis=2)
            A_B = tf.concat([A_B, tf.matmul(tf.concat([h_A_4, h_B_4], 1), W_interaction) + b_tensor], axis=-1)
            fully_size = 2 * tensor_size

    if pairwise_node_comparison:
        with tf.name_scope('pairwise_node_comparison'):
            # def sgm(x):
            #     return 1/(1+np.exp(-x))
            def hist(a,b):
                return tf.histogram_fixed_width(tf.nn.sigmoid(a@tf.transpose(b)), [0.0,1.0], nbins=n_bins)

            atomA=tf.dynamic_partition(outputA,graphA_topology.get_membership_placeholder(),batch_size)
            atomB=tf.dynamic_partition(outputB,graphB_topology.get_membership_placeholder(),batch_size)
            res=[]
            for i in range(batch_size):
                res.append(hist(atomA[i],atomB[i]))
            res=tf.to_float(tf.convert_to_tensor(res))
            res=tf.stop_gradient(res)
            A_B=tf.concat([A_B,res],-1)
            fully_size+=n_bins

    # A_B_interaction = h_A_4 + h_B_4
    # A_B_interaction = tf.concat([h_A_3, h_B_3], 1)
    # A_B = tf.nn.dropout(A_B, 1 - dropout3 * training)
    # interaction_size = 2 * Dense_size
    # interaction_size = Dense_size*Dense_size
    # interaction_size = (conv_size+Dense_size)*2
    # w_0 = tf.get_variable("w_0", [interaction_size, hidden_size], dtype=tf.float32)
    # w_0 = tf.get_variable("w_0", shape=[interaction_size, hidden_size], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
    # w_0 = tf.Variable(name="w_0", initial_value=tf.truncated_normal([fully_size, hidden_size], stddev=0.01),
    #                   dtype=tf.float32)
    w_0 = tf.get_variable('w_0', shape=[fully_size, hidden_size], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())

    b_0 = tf.get_variable("b_0", [hidden_size], dtype=tf.float32)
    # w_1 = tf.get_variable("w_1", [hidden_size, 2], dtype=tf.float32)
    # w_1 = tf.get_variable("w_1", shape=[hidden_size, 2], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
    # w_1 = tf.Variable(name="w_1", initial_value=tf.truncated_normal([hidden_size, 2], stddev=0.01),
    #                   dtype=tf.float32)
    w_1 = tf.get_variable('w_1', shape=[hidden_size, 2], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())

    b_1 = tf.get_variable("b_1", [2], dtype=tf.float32)

    # w_2 = tf.Variable(name="w_2", initial_value=tf.truncated_normal([2, 1], stddev=0.1), dtype=tf.float32)
    # b_2 = tf.get_variable("b_2", [1], dtype=tf.float32)

    logits = tf.matmul(A_B, w_0) + b_0
    logits = tf.nn.relu(logits)

    logits = tf.contrib.layers.batch_norm(logits, scope='bn', decay=0.9)
    # logits = tf.nn.dropout(logits, 1 - dropout4 * training)
    logits = tf.matmul(logits, w_1) + b_1
    # logits = tf.nn.relu(logits)
    # logits = tf.matmul(logits, w_2) + b_2
    # logits = tf.nn.dropout(logits, 1 - dropout5 * training)
    prob = tf.nn.softmax(logits)
    pred = prob[:,1]

    gold_matrix = tf.one_hot(label_gold, n_class, dtype=tf.float32)
    # label_gold=tf.cast(label_gold,tf.float32)
    # gold_matrix=tf.reshape(label_gold,[-1,1])
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gold_matrix))
    regularizers = tf.nn.l2_loss(w_0) + tf.nn.l2_loss(w_1)
    loss = loss + beta * regularizers
    # loss = -tf.reduce_mean(gold_matrix* tf.log(prob))
    # correct=tf.reshape(tf.equal(prob//0.5,gold_matrix),[-1])
    correct = tf.nn.in_top_k(logits, label_gold, 1)
    # eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))

    # predictions = prob//0.5
    predictions=tf.argmax(prob, 1)
    # label_gold_=tf.reshape(label_gold,[-1,1])
    # tp=tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(label_gold,1),tf.equal(predictions,1)),tf.float32))
    # tn=tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(label_gold,0),tf.equal(predictions,0)),tf.float32))
    # fp=tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(label_gold,0),tf.equal(predictions,1)),tf.float32))
    # fn=tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(label_gold,1),tf.equal(predictions,0)),tf.float32))


    if optimize_type == 'adadelta':
        clipper = 50
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        tvars = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
        loss = loss + lambda_l2 * l2_loss
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), clipper)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

    elif optimize_type == 'sgd':
        global_step = tf.Variable(0, name='global_step',
                                  trainable=False)  # Create a variable to track the global step.
        min_lr = 0.000001
        _lr_rate = tf.maximum(min_lr, tf.train.exponential_decay(learning_rate, global_step, 30000, 0.98))
        train_op = tf.train.GradientDescentOptimizer(learning_rate=_lr_rate).minimize(loss)

    elif optimize_type == 'adam':
        # clipper = 50
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=.9, beta2=.999, epsilon=1e-7)
        # tvars = tf.trainable_variables()
        # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
        # loss = loss + lambda_l2 * l2_loss
        # grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), clipper)
        train_op = optimizer.minimize(loss, global_step=global_step)


    elif optimize_type == 'ema':
        tvars = tf.trainable_variables()
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        # Create an ExponentialMovingAverage object
        ema = tf.train.ExponentialMovingAverage(decay=0.9999)

        maintain_averages_op = ema.apply(tvars)

        with tf.control_dependencies([train_op]):
            train_op = tf.group(maintain_averages_op)

    ########################################################################################################################
    # data loader
    tasks, datasets = load_interaction_data(filename=dataset, frac_train=frac_train)
    train_dataset, valid_dataset, test_dataset = datasets
    train_A, train_B = train_dataset
    valid_A, valid_B = valid_dataset
    test_dataset_A, test_dataset_B = test_dataset

    # a_valid = valid_A.iterbatches(batch_size, epoch=max_epochs, pad_batches=True, deterministic=True)
    # b_valid = valid_B.iterbatches(batch_size, epoch=max_epochs, pad_batches=True, deterministic=True)
    # evaluate_valid = [a_valid, b_valid]
    x_shape, y_shape, w_shape, i_shape = valid_A.get_shape()
    valid_step = int(x_shape[0] / batch_size) + 1
    lose_num_valid = valid_step * batch_size - x_shape[0]

    # a_test = test_dataset_A.iterbatches(batch_size, epoch=max_epochs+2, pad_batches=True, deterministic=True)
    # b_test = test_dataset_B.iterbatches(batch_size, epoch=max_epochs+2, pad_batches=True, deterministic=True)
    # evaluate_test = [a_test, b_test]
    x_shape, y_shape, w_shape, i_shape = test_dataset_A.get_shape()
    test_step = int(x_shape[0] / batch_size) + 1
    lose_num_test = test_step * batch_size - x_shape[0]

    # a_train = train_A.iterbatches(batch_size, epoch=max_epochs, pad_batches=True, deterministic=True)
    # b_train = train_B.iterbatches(batch_size, epoch=max_epochs, pad_batches=True, deterministic=True)
    # evaluate_train = [a_train, b_train]
    x_shape, y_shape, w_shape, i_shape = train_A.get_shape()
    train_step = int(x_shape[0] / batch_size) + 1
    lose_num_train = train_step * batch_size - x_shape[0]
    # max_step = train_step * max_epochs

    # initialize
    initializer = tf.global_variables_initializer()
    vars_ = {}
    for var in tf.global_variables():
        # print(var)
        vars_[var.name.split(":")[0]] = var
    saver = tf.train.Saver(vars_)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session()
    sess.run(initializer)
    if has_pre_trained_model:
        print("Restoring model from " + restore_path)
        saver.restore(sess, restore_path)
        print("DONE!")

    # training precessing

    total_loss = 0
    start_time = time.time()


    if mode=='training':

        def gen_dict(a_train,b_train,training_value=0):
            X_a, y_a, w_a, ids_a = a_train
            X_b, y_b, w_b, ids_b = b_train

            atoms_dictA = graphA_topology.batch_to_feed_dict(X_a)
            atoms_dictB = graphB_topology.batch_to_feed_dict(X_b)
            atoms_dict = atoms_dictA.copy()
            atoms_dict.update(atoms_dictB)

            Y = np.reshape(y_a, [batch_size])
            label_dict = {label_gold: Y}
            training_dict = {training: training_value}
            feed_dict=merge_dicts([label_dict, atoms_dict, training_dict])
            return feed_dict

        accuracy, auc, evlt = evaluate(test_dataset_A,test_dataset_B,test_step, lose_num_test)
        print("Accuracy for test set is", accuracy, auc, evlt)
        for epoch in range(max_epochs):
            a_train = train_A.iterbatches(batch_size, pad_batches=True, deterministic=True)
            b_train = train_B.iterbatches(batch_size, pad_batches=True, deterministic=True)
            print('epoch %d' % (epoch))
            for i in range(train_step):
                a=next(a_train)
                b=next(b_train)
                _, loss_value = sess.run([train_op, loss], feed_dict=gen_dict(a,b,1))
                total_loss += loss_value
                if reverse:
                    _, loss_value = sess.run([train_op, loss], feed_dict=gen_dict(b,a,1))
                    total_loss += loss_value
                if (i+1) == train_step:
                    duration = time.time() - start_time
                    start_time = time.time()
                    print('Step %d: loss = %.2f (%.3f sec)' % (i, total_loss, duration))
                    losses = total_loss
                    total_loss = 0.0
                    # Evaluate against the validation set.+
                    #accuracy = evaluate(graphA_topology, graphB_topology, label_gold, training, sess, evaluate_train, batch_size, train_step, lose_num_train, mode='prediction')
                    #print("Accuracy for train set is %.2f" % accuracy)

                    if (epoch%5) == 4 or losses <= 1000:
                        print('Validation Data Eval:')
                        accuracy_valid, auc_valid, evlt_valid = evaluate(valid_A,valid_B, valid_step, lose_num_valid)
                        print("Current accuracy is " ,accuracy_valid, auc_valid, evlt_valid)
                        # saver.restore(sess, best_path)  # best_model
                        accuracy ,auc,evlt= evaluate(test_dataset_A,test_dataset_B,test_step, lose_num_test)
                        print("Accuracy for test set is" , accuracy ,auc,evlt)

                        if accuracy_valid >= best_accuracy:
                            best_accuracy = accuracy_valid
                            saver.save(sess, best_path)
                            print('model saved in',best_path)

        #
        print("Best accuracy on dev set is %.2f" % best_accuracy)
        # decoding
        print('Decoding on the test set:')
        saver.restore(sess, best_path)# best_model
        accuracy ,auc,evlt= evaluate(test_dataset_A,test_dataset_B,test_step, lose_num_test)
        print("Accuracy for test set is" , accuracy ,auc,evlt)

    # elif mode=='prediction':
    #
    #     X_a, y_a, w_a, ids_a = a_train
    #     X_b, y_b, w_b, ids_b = b_train
    #
    #     atoms_dictA = graphA_topology.batch_to_feed_dict(X_a)
    #     atoms_dictB = graphB_topology.batch_to_feed_dict(X_b)
    #     atoms_dict = atoms_dictA.copy()
    #     atoms_dict.update(atoms_dictB)
    #
    #     Y = np.reshape(y_a, [batch_size])
    #     label_dict = {label_gold: Y}
    #     training_dict = {training: 0}
    #     feed_dict = merge_dicts([label_dict, atoms_dict, training_dict])
    #
    #     result = sess.run(predictions, feed_dict=feed_dict)
    #     print(result)
    # elif mode=='prediction_vector':
    #     print("Restoring model from " + restore_path)
    #     saver.restore(sess, restore_path)
    #     print("DONE!")
    #
    #     a_train = list(train_A.iterbatches(batch_size, pad_batches=True, deterministic=True))
    #     b_train = list(train_B.iterbatches(batch_size, pad_batches=True, deterministic=True))
    #     evaluate_train = [a_train, b_train]
    #     x_shape, y_shape, w_shape, i_shape = train_A.get_shape()
    #     train_step = int(x_shape[0] / batch_size) + 1
    #     lose_num_train = train_step * batch_size - x_shape[0]
    #     max_step = train_step * max_epochs
    #
    #     train_data=[],[],[]
    #     for i in range(train_step):
    #         X_a, y_a, w_a, ids_a = a_train[i]
    #         X_b, y_b, w_b, ids_b = b_train[i]
    #
    #         atoms_dictA = graphA_topology.batch_to_feed_dict(X_a)
    #         atoms_dictB = graphB_topology.batch_to_feed_dict(X_b)
    #         atoms_dict = atoms_dictA.copy()
    #         atoms_dict.update(atoms_dictB)
    #
    #         Y = np.reshape(y_a, [batch_size])
    #         label_dict = {label_gold: Y}
    #         training_dict = {training: 1}
    #         feed_dict = merge_dicts([label_dict, atoms_dict, training_dict])
    #
    #         A_vecs, B_vecs = sess.run([h_A_4, h_B_4], feed_dict=feed_dict)
    #         train_data[0].append(A_vecs)
    #         train_data[1].append(B_vecs)
    #         train_data[2].append(y_a)
    #
    #     a_valid = list(valid_A.iterbatches(batch_size, pad_batches=True, deterministic=True))
    #     b_valid = list(valid_B.iterbatches(batch_size, pad_batches=True, deterministic=True))
    #     # evaluate_test = [a_test, b_test]
    #     x_shape, y_shape, w_shape, i_shape = valid_A.get_shape()
    #     valid_step = int(x_shape[0] / batch_size) + 1
    #     # lose_num_test = test_step * batch_size - x_shape[0]
    #
    #     valid_vectors = [], [], []
    #     for i in range(valid_step):
    #         X_a, y_a, w_a, ids_a = a_valid[i]
    #         X_b, y_b, w_b, ids_b = b_valid[i]
    #
    #         atoms_dictA = graphA_topology.batch_to_feed_dict(X_a)
    #         atoms_dictB = graphB_topology.batch_to_feed_dict(X_b)
    #         atoms_dict = atoms_dictA.copy()
    #         atoms_dict.update(atoms_dictB)
    #
    #         Y = np.reshape(y_a, [batch_size])
    #         label_dict = {label_gold: Y}
    #         training_dict = {training: 0}
    #         feed_dict = merge_dicts([label_dict, atoms_dict, training_dict])
    #
    #         A_vecs, B_vecs = sess.run([h_A_4, h_B_4], feed_dict=feed_dict)
    #         valid_vectors[0].append(A_vecs)
    #         valid_vectors[1].append(B_vecs)
    #         valid_vectors[2].append(y_a)
    #
    #     a_test = list(test_dataset_A.iterbatches(batch_size, pad_batches=True, deterministic=True))
    #     b_test = list(test_dataset_B.iterbatches(batch_size, pad_batches=True, deterministic=True))
    #     # evaluate_test = [a_test, b_test]
    #     x_shape, y_shape, w_shape, i_shape = test_dataset_A.get_shape()
    #     test_step = int(x_shape[0] / batch_size) + 1
    #     # lose_num_test = test_step * batch_size - x_shape[0]
    #
    #     vectors=[],[],[]
    #     for i in range(test_step):
    #         X_a, y_a, w_a, ids_a = a_test[i]
    #         X_b, y_b, w_b, ids_b = b_test[i]
    #
    #         atoms_dictA = graphA_topology.batch_to_feed_dict(X_a)
    #         atoms_dictB = graphB_topology.batch_to_feed_dict(X_b)
    #         atoms_dict = atoms_dictA.copy()
    #         atoms_dict.update(atoms_dictB)
    #
    #         Y = np.reshape(y_a, [batch_size])
    #         label_dict = {label_gold: Y}
    #         training_dict = {training: 0}
    #         feed_dict = merge_dicts([label_dict, atoms_dict, training_dict])
    #
    #         A_vecs,B_vecs=sess.run([h_A_4,h_B_4],feed_dict=feed_dict)
    #         vectors[0].append(A_vecs)
    #         vectors[1].append(B_vecs)
    #         vectors[2].append(y_a)
    #     import pickle
    #     with open('../datasets/vectors_{}.pkl'.format(filename), 'wb') as w_file:
    #         pickle.dump(train_data,w_file)
    #         pickle.dump(valid_vectors,w_file)
    #         pickle.dump(vectors, w_file)

