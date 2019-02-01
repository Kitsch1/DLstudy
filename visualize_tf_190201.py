import tensorflow as tf
import matplotlib.pypolt as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# solution of vanishing gradient problem
def loss(y,t):
    cross_entropy=\
        tf.reduce_mean(
            -tf.reduce_sum(
                t*tf.log(tf.clip_by_value(y,1e-10,1.0)),
                reduction_indices=[1]))
    return cross_entropy


# Separate validation data from training data : start

N_train = 20000
N_validation = 4000

X_train, X_test, Y_train, Y_test = \
         train_test_split(X,Y,train_size=N_train)

X_train, X_validation, Y_train, Y_validation = \
         train_test_split(X_train,Y_train,test_size = N_validation)

# Separate validation data from training data : end
# Validation data is used to evaluate the learning of model

# Evaluate accuracy each epoch
epochs = 50
batch_size = 100
n_batches = N_train // batch_size

history = {
    'val_loss': [],
    'val_acc': []
    }

for epoch in range(epochs):
    X_, Y_ = shuffle(X_train, Y_train)

    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size

        sess.run(train_step, feed_dict={
            x: X_[start:end],
            t: Y_[start:end],
            keep_prob: p_keep
        })

        # evaluate by validation data
        val_loss = loss.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation,
            keep_prob: 1.0
        })
        val_acc = accuracy.eval(session=sess, feed_dict={
            x: X_validation,
            t: Y_validation,
            keep_prob: 1.0
        })

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

# drawing graph

plt.rc('font',family='serif')
fig = plt.figure() # prepare graph

# drawing graph using data
plt.plot(range(epochs),history['val_acc'],label='acc',color='black')

# label name
plt.xlabel('epochs')
plt.ylabel('validation loss')

# showing graph
plt.show()
#plt.savefig('mnist_tensorflow.eps')

# drawing graph (accuracy and loss)

fig = plt.figure()

ax_acc = fig.add_subplot(111) # accuracy label
ax_acc.plot(range(epochs),history['val_acc'],label='acc',color='black')

ax_loss = ax_acc.twinx() # loss
ax_loss.plot(range(epochs),history['val_loss'],label='loss',color='gray')

plt.xlabel('epochs')
plt.show()
#plt.savefig('mnist_tensorflow.eps')



