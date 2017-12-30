import numpy as np
import time
from SqueezeNet import *


def loss_generate_SqueezeNet_SH(data_train, label_train, data_val, label_val, r, model_i, epoch, es_flag, best_accuracy,
                                best_loss, weight_decay, learning_rate, batch_size, dr, ds, print_every, model_print,
                                epoch_print, save_print, overall_print, first_train, validation):
    '''
    Trial one model several times and generate loss for Successive Haliving Algorithm.

    :param data_train: training data
    :param label_train: training labels
    :param data_val: validation data
    :param label_val: validation labels
    :param r: trial times
    :param model_i: the previouus i-th model need to be load in
    :param epoch: previous number of epoches that the model has been trained
    :param es_flag: whether detect an early stopping condition
    :param best_accuracy: the accuracy recorded at the last stopping-check
    :param best_loss: the loss recorded at the last stopping-check
    :param weight_decay: weight decay
    :param learning_rate: learning rate
    :param batch_size: batch size
    :param dr: decay rate
    :param ds: decay steps
    :param print_every: print information every certain steps
    :param model_print: whether print model information at the begining
    :param epoch_print: whether print information after each training / validation epoch
    :param save_print: whether print saving information in the end
    :param overall_print: whether print overall information
    :param first_train: whether first train process or not, if false, load the previous saved model
    :param validation: the frequency of validation, usually 5 (epoches)

    :return: val_loss (validation loss), val_accuracy (validation accuracy), epoch (after finishing this subroutine, the number of total
    epoches has been passed through), best_loss, best_accuracy (loss and accuracy recorded at the last stopping-check)
    '''

    import tensorflow as tf
    import numpy as np

    epoch = int(epoch)
    val_loss = 0
    val_accuracy = 0
    save_path = ''
    tf.reset_default_graph()
    with tf.Session() as sess:
        if first_train:
            model = SqueezeNet(weight_decay=weight_decay)
            mean_loss = model.loss

            global_step = tf.Variable(0, trainable=False)
            starter_lr = learning_rate
            lr = tf.train.exponential_decay(starter_lr, global_step, ds, dr, staircase=True)
            optimizer = tf.train.MomentumOptimizer(lr, 0.9)
            train_step = optimizer.minimize(mean_loss)

            correct_prediction = tf.equal(tf.cast(tf.argmax(model.classifier, 1), tf.int32), model.labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
        else:
            if es_flag and epoch % validation != 0:
                save_path = "./models/my_SqueezeNet_model" + str(model_i) + "_temp.ckpt"
            else:
                save_path = "./models/my_SqueezeNet_model" + str(model_i) + ".ckpt"
            model = SqueezeNet(weight_decay=weight_decay)
            mean_loss = model.loss

            global_step = tf.Variable(0, trainable=False)
            starter_lr = learning_rate
            lr = tf.train.exponential_decay(starter_lr, global_step, ds, dr, staircase=True)
            optimizer = tf.train.MomentumOptimizer(lr, 0.9)
            train_step = optimizer.minimize(mean_loss)

            correct_prediction = tf.equal(tf.cast(tf.argmax(model.classifier, 1), tf.int32), model.labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            tf.global_variables_initializer()
            saver = tf.train.Saver()
            saver.restore(sess, save_path)

        if model_print:
            print('weight_decay: {0}; learning_rate: {1}; batch_size: {2}; decay_rate: {3}; decay_step: {4} ......' \
                  .format(weight_decay, starter_lr, batch_size, dr, ds))

        # counter
        iter_cnt = 0
        tr_epochs = int(np.ceil(r / np.ceil(data_train.shape[0] / batch_size)))

        for e in range(tr_epochs):
            # shuffle the dataset; important!
            train_indices = np.arange(data_train.shape[0])
            np.random.shuffle(train_indices)

            # keep track of losses and accuracy
            epochs_losses = []
            epochs_corrects = 0
            # make sure we iterate over the dataset once despite the last epochs
            if e == tr_epochs - 1 and r % np.ceil(data_train.shape[0] / batch_size):
                upper = (r - np.ceil(data_train.shape[0] / batch_size) * e) * batch_size
            else:
                upper = data_train.shape[0]

            for l in range(int(np.ceil(upper / batch_size))):
                # generate indices for the batch
                start_idx = (l * batch_size) % upper
                idx = train_indices[int(start_idx): int(start_idx + batch_size)]
                # get batch size
                actual_batch_size = label_train[idx].shape[0]
                # have tensorflow compute loss and correct predictions and (if given) perform a training step
                loss, ac, _ = sess.run([mean_loss, accuracy, train_step],
                                       feed_dict={model.image: data_train[idx, :],
                                                  model.labels: list(label_train[idx])})
                # aggregate performance stats
                epochs_losses.append(loss * actual_batch_size)
                epochs_corrects += ac * actual_batch_size
                # print every now and then
                if print_every != 0 and (iter_cnt % print_every) == 0:
                    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                          .format(iter_cnt, loss, ac))
                iter_cnt += 1

            epochs_accuracy = epochs_corrects / upper
            epochs_loss = np.sum(epochs_losses) / upper

            if epoch_print:
                print("Training epoch {2}, loss = {0:.3g} and accuracy = {1:.3g}" \
                      .format(epochs_loss, epochs_accuracy, epoch + 1))

            if (epoch + 1) % validation == 0 or e == tr_epochs - 1:
                val_indices = np.arange(data_val.shape[0])
                np.random.shuffle(val_indices)
                # keep track of losses and accuracy
                val_losses = []
                corrects = 0
                # make sure we iterate over the dataset once
                for l in range(int(np.ceil(data_val.shape[0] / batch_size))):
                    # generate indices for the batch
                    start_idx = (l * batch_size) % data_val.shape[0]
                    idx = val_indices[int(start_idx): int(start_idx + batch_size)]
                    # get batch size
                    actual_batch_size = label_val[idx].shape[0]
                    # have tensorflow compute loss and correct predictions and (if given) perform a training step
                    variables = [mean_loss, accuracy]
                    loss, ac = sess.run(variables,
                                        feed_dict={model.image: data_val[idx, :], model.labels: list(label_val[idx])})
                    # aggregate performance stats
                    val_losses.append(loss * actual_batch_size)
                    corrects += ac * actual_batch_size

                val_accuracy = corrects / data_val.shape[0]
                val_loss = np.sum(val_losses) / data_val.shape[0]

                if epoch_print:
                    print("Validation epoch {2}, loss = {0:.3g} and accuracy = {1:.3g}" \
                          .format(val_loss, val_accuracy, epoch + 1))

                # deal with the early stopping
                if (epoch + 1) % validation == 0:
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_loss = val_loss
                        es_flag = 0
                        # save the model in case that the early stopping happens
                        save_path = saver.save(sess, "./models/my_SqueezeNet_model" + str(model_i) + ".ckpt")
                        if save_print:
                            print("save model:{0} Finished".format(save_path))
                    else:  # stop, and we don't need to save the model at this moment
                        val_loss = best_loss
                        val_accuracy = best_accuracy
                        epoch += 1
                        es_flag = 1
                        if overall_print:
                            print('detect an early stopping condition!')
                            print("Overall loss = {0:.3g} and accuracy = {1:.3g}" \
                                  .format(val_loss, val_accuracy))
                        break

                if e == tr_epochs - 1:
                    if overall_print:
                        print("Overall loss = {0:.3g} and accuracy = {1:.3g}" \
                              .format(val_loss, val_accuracy))
                    if es_flag:
                        save_path = saver.save(sess, "./models/my_SqueezeNet_model" + str(model_i) + "_temp.ckpt")
                    elif (epoch + 1) % validation != 0:
                        save_path = saver.save(sess, "./models/my_SqueezeNet_model" + str(model_i) + ".ckpt")

                    if save_print:
                        print("save model:{0} Finished".format(save_path))
            epoch += 1

    return val_loss, val_accuracy, epoch, best_loss, best_accuracy, es_flag



def test_SqueezeNet(data_test, label_test, model_i, weight_decay, learning_rate, batch_size, dr, ds, print_every):
    '''
    Test the model at the final step of hyperparameters optimization

    :param data_test
    :param label_test
    :param model_i: the previouus i-th model need to be load in
    :param weight_decay: weight decay
    :param learning_rate: learning rate
    :param batch_size: batch size
    :param dr: decay rate
    :param ds: decay steps
    :param print_every: print information every certain steps

    :return: test_loss, test_accuracy
    '''

    import tensorflow as tf
    import numpy as np

    tf.reset_default_graph()
    with tf.Session() as sess:
        save_path = "./models/my_SqueezeNet_model" + str(model_i) + ".ckpt"
        model = SqueezeNet(weight_decay=weight_decay)
        mean_loss = model.loss
        starter_lr = learning_rate

        correct_prediction = tf.equal(tf.cast(tf.argmax(model.classifier, 1), tf.int32), model.labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.global_variables_initializer()
        saver = tf.train.Saver()
        saver.restore(sess, save_path)

        print('Test: ')
        print('weight_decay: {0}; learning_rate: {1}; batch_size: {2}; decay_rate: {3}; decay_step: {4} ......' \
              .format(weight_decay, starter_lr, batch_size, dr, ds))

        test_indices = np.arange(data_test.shape[0])
        np.random.shuffle(test_indices)

        # keep track of losses and accuracy
        test_losses = []
        corrects = 0
        iter_cnt = 0
        # make sure we iterate over the dataset once
        for l in range(int(np.ceil(data_test.shape[0] / batch_size))):
            # generate indices for the batch
            start_idx = (l * batch_size) % data_test.shape[0]
            idx = test_indices[int(start_idx): int(start_idx + batch_size)]
            # get batch size
            actual_batch_size = label_test[idx].shape[0]
            # have tensorflow compute loss and correct predictions and (if given) perform a training step
            variables = [mean_loss, accuracy]
            loss, ac = sess.run(variables,
                                feed_dict={model.image: data_test[idx, :], model.labels: list(label_test[idx])})
            # aggregate performance stats
            test_losses.append(loss * actual_batch_size)
            corrects += ac * actual_batch_size
            # print every now and then
            if print_every != 0 and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch test loss = {1:.3g} and accuracy of {2:.2g}" \
                      .format(iter_cnt, loss, ac))
            iter_cnt += 1

        test_accuracy = corrects / data_test.shape[0]
        test_loss = np.sum(test_losses) / data_test.shape[0]
        print("Test loss = {0:.3g} and accuracy = {1:.3g}".format(test_loss, test_accuracy))

    return test_loss, test_accuracy


def get_param_comb(j, param_len):
    '''
    Get the combination of parameters from given index j
    '''
    k = len(param_len)
    idx = np.zeros([k, ])
    idx[k - 1] = j % param_len[k - 1]
    a = list(range(k))
    a.reverse()
    for i in a:
        j /= param_len[i]
        idx[i - 1] = np.floor(j) % param_len[i - 1]
    return idx


def SH_hyperopt(N, K, loss_generate, data_train, label_train, data_val, label_val, data_test, label_test,
                param_dict, print_every, val_freq, pretrained):
    '''
    - Successive-Halving algorithm for hyperparameters optimization
    - Use SqueezeNet as base model

    :param N: Budget;
    :param K: Number of parameter-combinations (arms);
    :param loss_generate: Method to generate losses; (import loss_generate_SqueezeNet from my_SqueezeNet)
    :param data_train: Training data;
    :param label_train: Training labels;
    :param data_val: Validation data;
    :param label_val: Validation labels;
    :param data_test
    :param label_test
    :param param_dict: A dictionary of 5 parameters (weight_decay, learning_rate, batch_size, dr, ds) to be optimized;
    :param print_every: How often to print information.
    :param val_freq: The frequency of validation, usually 5 (epoches)
    :param pretrained: Whether this subroutine is called in S3BA (the model has been pretrained by S3BA)

    :return: S_k[0] (the best arm), val_loss, val_accuracy, test_loss, test_accuracy, runtime, epoches,
    es_flag(whether detect an early stopping condition)
    '''
    R = 0
    S_k = np.arange(1, K + 1)
    size_Sk = len(S_k)
    r = 0
    s = size_Sk

    param_len = np.zeros([len(param_dict), ]) # len(param_dict) = 5 by default
    weight_decay, learning_rate, batch_size, dr, ds = param_dict['weight_decay'], param_dict['learning_rate'], \
                                                      param_dict['batch_size'], param_dict['decay_rate'], \
                                                      param_dict['decay_step']
    param_len[0], param_len[1], param_len[2], param_len[3], param_len[4] = len(weight_decay), len(learning_rate), len(
        batch_size), len(dr), len(ds)
    param_comb = np.zeros([5, ])
    epoches = np.zeros([K, ])
    es_flag = np.zeros([K, ])
    best_accuracy = np.zeros([K, ])
    best_loss = np.zeros([K, ])
    val_loss_min, ac = 100 ** 100, 0
    time_start = time.time()

    for k in np.arange(np.ceil(np.log2(K))):
        print('Round ', int(k))
        r = np.floor(N / (size_Sk * np.ceil(np.log2(K))))  # pull each arm for r times
        i = 0
        l = np.zeros([size_Sk, 1])
        for j in S_k:
            j = int(j)
            param_comb = get_param_comb(j-1, param_len).astype('int32') # parameter combination corresponding to certain model (arm) j

            if k == 0 and pretrained == False: # hasn't been pretrained (first train)
                if k == np.ceil(np.log2(K)) - 1:
                    l[i], accuracy, epoches[j - 1], best_loss[j - 1], best_accuracy[j - 1], es_flag[j - 1] = \
                        loss_generate(data_train, label_train, data_val, label_val, r=r, model_i=j, epoch=epoches[j - 1],
                                      es_flag=es_flag[j - 1], best_accuracy=best_accuracy[j - 1], best_loss=best_loss[j - 1],
                                      weight_decay=weight_decay[param_comb[0]], learning_rate=learning_rate[param_comb[1]],
                                      batch_size=batch_size[param_comb[2]], dr=dr[param_comb[3]], ds=ds[param_comb[4]],
                                      print_every=print_every, model_print=True, epoch_print=True, save_print=True,
                                      overall_print=True, first_train=True, validation=val_freq)
                    if es_flag[j - 1]:
                        l[i], accuracy = best_loss[j - 1], best_accuracy[j - 1]

                    if val_loss_min > l[i]:
                        val_loss_min = l[i]
                        ac = accuracy
                else:
                    l[i], _, epoches[j - 1], best_loss[j - 1], best_accuracy[j - 1], es_flag[j - 1] = \
                        loss_generate(data_train, label_train, data_val, label_val, r=r, model_i=j, epoch=epoches[j - 1],
                                      es_flag=es_flag[j - 1], best_accuracy=best_accuracy[j - 1], best_loss=best_loss[j - 1],
                                      weight_decay=weight_decay[param_comb[0]], learning_rate=learning_rate[param_comb[1]],
                                      batch_size=batch_size[param_comb[2]], dr=dr[param_comb[3]], ds=ds[param_comb[4]],
                                      print_every=print_every, model_print=True, epoch_print=True, save_print=True,
                                      overall_print=True, first_train=True, validation=val_freq)
                    if es_flag[j - 1]:
                        l[i] = best_loss[j - 1]

            elif k == 0 and pretrained:
                if k == np.ceil(np.log2(K)) - 1:
                    l[i], accuracy, epoches[j - 1], best_loss[j - 1], best_accuracy[j - 1], es_flag[j - 1] = \
                        loss_generate(data_train, label_train, data_val, label_val, r=r, model_i=j,
                                      epoch=epoches[j - 1], es_flag=es_flag[j - 1], best_accuracy=best_accuracy[j - 1],
                                      best_loss=best_loss[j - 1], weight_decay=weight_decay[param_comb[0]],
                                      learning_rate=learning_rate[param_comb[1]], batch_size=batch_size[param_comb[2]],
                                      dr=dr[param_comb[3]], ds=ds[param_comb[4]], print_every=print_every, model_print=True,
                                      epoch_print=True, save_print=True, overall_print=True, first_train=False, validation=val_freq)
                    if es_flag[j - 1]:
                        l[i], accuracy = best_loss[j - 1], best_accuracy[j - 1]

                    if val_loss_min > l[i]:
                        val_loss_min = l[i]
                        ac = accuracy
                else:
                    l[i], _, epoches[j - 1], best_loss[j - 1], best_accuracy[j - 1], es_flag[j - 1] = \
                        loss_generate(data_train, label_train, data_val, label_val, r=r, model_i=j,
                                      epoch=epoches[j - 1], es_flag=es_flag[j - 1], best_accuracy=best_accuracy[j - 1],
                                      best_loss=best_loss[j - 1], weight_decay=weight_decay[param_comb[0]],
                                      learning_rate=learning_rate[param_comb[1]], batch_size=batch_size[param_comb[2]],
                                      dr=dr[param_comb[3]], ds=ds[param_comb[4]], print_every=print_every, model_print=True,
                                      epoch_print=True, save_print=True, overall_print=True, first_train=False, validation=val_freq)
                    if es_flag[j - 1]:
                        l[i] = best_loss[j - 1]

            elif k != 0 and k == np.ceil(np.log2(K)) - 1:
                l[i], accuracy, epoches[j - 1], best_loss[j - 1], best_accuracy[j - 1], es_flag[j - 1] = \
                    loss_generate(data_train, label_train, data_val, label_val, r=r, model_i=j, epoch=epoches[j - 1],
                                  es_flag=es_flag[j - 1], best_accuracy=best_accuracy[j - 1], best_loss=best_loss[j - 1],
                                  weight_decay=weight_decay[param_comb[0]], learning_rate=learning_rate[param_comb[1]],
                                  batch_size=batch_size[param_comb[2]], dr=dr[param_comb[3]], ds=ds[param_comb[4]],
                                  print_every=print_every, model_print=True, epoch_print=True, save_print=True,
                                  overall_print=True, first_train=False, validation=val_freq)
                if es_flag[j - 1]:
                    l[i], accuracy = best_loss[j - 1], best_accuracy[j - 1]

                if val_loss_min > l[i]:
                    val_loss_min = l[i]
                    ac = accuracy

            else:
                l[i], _, epoches[j - 1], best_loss[j - 1], best_accuracy[j - 1], es_flag[j - 1] = \
                    loss_generate(data_train, label_train, data_val, label_val, r=r, model_i=j, epoch=epoches[j - 1],
                                  es_flag=es_flag[j - 1], best_accuracy=best_accuracy[j - 1], best_loss=best_loss[j - 1],
                                  weight_decay=weight_decay[param_comb[0]], learning_rate=learning_rate[param_comb[1]],
                                  batch_size=batch_size[param_comb[2]], dr=dr[param_comb[3]], ds=ds[param_comb[4]],
                                  print_every=print_every, model_print=True, epoch_print=True, save_print=True,
                                  overall_print=True, first_train=False,  validation=val_freq)
                if es_flag[j - 1]:
                    l[i] = best_loss[j - 1]

            i += 1

        l_dict = {}  ## build a dictionary in form of {arm: loss}
        i = 0
        for j in S_k:
            l_dict[j] = l[i]
            i += 1

        R += int(size_Sk) * r
        R = int(R)

        l_sorted = sorted(l_dict.items(), key=lambda d: d[1], reverse=False)  ## sorted on loss
        sigma = np.zeros([len(l_sorted), 1])  ## labels of arms corresponding to the ascend losses
        for i in np.arange(len(l_sorted)):
            sigma[i] = l_sorted[i][0]

        size_Sk = np.ceil(len(sigma) / 2)  ## renew size of S_k
        size_Sk = int(size_Sk)

        S_k = sigma[np.arange(size_Sk)]  ## renew S_k
        S_k = S_k.reshape(S_k.shape[0]).astype(np.int32)
        print('now models to be chosen: ', list(S_k))

    val_loss = val_loss_min
    val_accuracy = ac
    time_end = time.time()
    runtime = time_end - time_start
    print('Total Runtime: ', runtime, ' s')

    param_comb = get_param_comb(S_k[0] - 1, param_len).astype('int32')
    test_loss, test_accuracy = test_SqueezeNet(data_test, label_test, model_i=S_k[0], weight_decay=weight_decay[param_comb[0]],
                                               learning_rate=learning_rate[param_comb[1]], batch_size=batch_size[param_comb[2]],
                                               dr=dr[param_comb[3]], ds=ds[param_comb[4]], print_every=0)

    return S_k[0], val_loss, val_accuracy, test_loss, test_accuracy, runtime, epoches, es_flag


