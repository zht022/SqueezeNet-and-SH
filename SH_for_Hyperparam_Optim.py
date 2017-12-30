import numpy as np
import time
from data_utils import *
from SqueezeNet import *
from hyperopt_method import *

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(49000, 1000, 10000)
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

'''
As an example, we tune 5 parameters: weight_decay, learning_rate, batch_size, decay_rate, and decay_step; and try trial
budgets N from {5000, 10000, 15000, ..., 100000}. We just show an example here, but believe that there exists better
tuning scheme.
'''
paramdict = {'weight_decay':[0.00005, 0.0001, 0.0005, 0.001, 0.005], 'learning_rate':[0.001, 0.005, 0.01],
             'batch_size':[128], 'decay_rate':[0.96], 'decay_step':[1000]}
N = np.arange(5000, 100000+1, 5000)
recommendation = np.zeros([len(N), ])
validation_loss = np.zeros([len(N), ])
validation_accuracy = np.zeros([len(N), ])
test_loss = np.zeros([len(N), ])
test_accuracy = np.zeros([len(N), ])
runtime = np.zeros([len(N), ])
epoches = np.zeros([len(N), 15])
es_flag = np.zeros([len(N), 15])
i = 16
for n in np.arange(5000, 100000, 5000):
    print('###############################################################################################################')
    print('Budgets = ', n)
    recommendation[i], validation_loss[i], validation_accuracy[i], test_loss[i], test_accuracy[i], runtime[i], epoches[i, :], es_flag[i, :] = \
        SH_hyperopt(N=n, K=15, loss_generate=loss_generate_SqueezeNet_SH, data_train=X_train, label_train=y_train,
                    data_val=X_val, label_val=y_val, data_test=X_test, label_test=y_test,
                    param_dict=paramdict, print_every=0, val_freq=10, pretrained=False)
    i += 1


print('###############################################################################################################')
print('Show results now ......')

print('The models be chosen at each step: ')
print(recommendation)

print('Validation loss of each step: ')
print(validation_loss)

print('Validation accuracy of each step: ')
print(validation_accuracy)

print('Runtime of each step: ')
print(runtime)
np.savetxt("SH_runtime.txt", runtime)

print('Total runtime: ', np.sum(runtime))

print('Epoches run for each step: ')
print(epoches)

print('Test loss of each step: ')
print(test_loss)

print('Test accuracy of each step: ')
print(test_accuracy)



