import math, random
from collections import defaultdict
import tensorflow as tf
import numpy as np
from comet_ml import Experiment



TF_CPP_MIN_LOG_LEVEL=2


def leaky_relu(x, alpha=0.01):
    activation = tf.maximum(x,alpha*x)
    return activation 

def discriminator(b,reuse=False, alpha=0.2, training=True):
    """Compute discriminator score for a batch of input bs.

    """
    with tf.variable_scope("discriminator"):
        fc1 = tf.layers.dense(inputs=b, units=256, activation=leaky_relu)
        fc2 = tf.layers.dense(inputs=fc1, units=256, activation=leaky_relu)
        logits = tf.layers.dense(inputs=fc2, units=1)        
        return logits

def generator(a):
    """Generate b' conditional on input a 
    """
    with tf.variable_scope("generator"):        
        fc1 = tf.layers.dense(inputs=a, units=1024, activation=tf.nn.relu)
        fc2 = tf.layers.dense(inputs=fc1, units=1024, activation=tf.nn.relu)
        out = tf.layers.dense(inputs=fc2, units=300*19, activation=tf.nn.tanh)
        return out


def gan_loss(logits_real, logits_fake):
    """Compute the GAN loss.
    
    Inputs:
    - logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Log probability that the b is real for each real b
    - logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Log probability that the b is real for each fake b
    
    Returns:
    - D_loss: discriminator loss scalar
    - G_loss: generator loss scalar
    """
    
    # Target label vector for generator loss and used in discriminator loss.
    true_labels = tf.ones_like(logits_fake)
    real_b_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=true_labels)
    fake_b_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=1-true_labels)
    
    # Combine and average losses over the batch
    D_loss = real_b_loss + fake_b_loss 
    D_loss = tf.reduce_mean(D_loss)
    
    # GENERATOR is trying to make the discriminator output 1 for all its output.
    # So we use our target label vector of ones for computing generator loss.
    G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=true_labels)
    
    # Average generator loss over the batch.
    G_loss = tf.reduce_mean(G_loss)
    
    return D_loss, G_loss


# create an AdamOptimizer for D_solver and G_solver
def get_solvers(learning_rate=1e-3, beta1=0.5):
    """Create solvers for GAN training.
    """
    
    D_solver = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    
    return D_solver, G_solver




def ids_to_text(ids, dictionary):
    return " ".join([dictionary.reverse(ids[i]) for i in range(len(ids))])


def dev_eval(A_dev, B_dev, batch_size, sess):
    epoch_G_loss = 0
    epoch_D_loss = 0
    total_examples = A_dev.shape[0]
    batches = int(total_examples / batch_size)
    examples = 0
    for i in (range(batches)):
        if i == batches - 1:
            x = A_dev[i * batch_size:]
            y = B_dev[i * batch_size:]
            if x.shape[0] != batch_size:
                continue
        else:
            x = A_dev[i * batch_size: (i + 1) * batch_size]
            y = B_dev[i * batch_size: (i + 1) * batch_size]
        examples += batch_size

        G_loss_curr, D_loss_curr = sess.run([G_loss, D_loss], feed_dict = {b: y,a: x})

        epoch_G_loss += G_loss_curr
        epoch_D_loss += D_loss_curr

    print("DEV:\t GLoss: {},\t DLoss: {}".format(epoch_G_loss / examples, epoch_D_loss / examples))
    experiment.log_metric("dev_G_loss", epoch_G_loss/ examples)
    experiment.log_metric("dev_D_loss",  epoch_D_loss / examples)


def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step,\
              show_every=250, print_every=2, batch_size=128, num_epoch=10,shuffle = False,\
              num_iterations = 200):
    """Train a GAN for a certain number of epochs.
    
    Inputs:
    - sess: A tf.Session that we want to use to run our data
    - G_train_step: A training step for the Generator
    - G_loss: Generator loss
    - D_train_step: A training step for the Generator
    - D_loss: Discriminator loss
    - G_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for generator
    - D_extra_step: A collection of tf.GraphKeys.UPDATE_OPS for discriminator
    Returns:
        Nothing
    """


    # A = np.load('A.npy') #, encoding = 'latin1'
    # A = A[:128*2]
    # A = A.astype(int)

    # B = np.random.rand(128*2,19) 
    # y = np.random.rand(128*2)
    # B = B.astype(int)


    num_train_examples = 100000
    num_dev_examples = batch_size*16

    embeddings_matrix = np.load('../../data/wm.npy')


    labels = np.load('../../data/labels.npy')

    A = np.load('../../data/A_14.npy') #shape [none, 14]
    A = A.astype(int)
    print(A.shape)
    A = A[~(labels == 0)]
    A = A[:num_train_examples + num_dev_examples,:]
    print("\n\n shapeof A", A.shape)

    B = np.load('../../data/B_19.npy', encoding = 'latin1')
    B = B.astype(int)

    B = B[~(labels == 0)]
    B = B[:num_train_examples + num_dev_examples,:]


    A = tf.nn.embedding_lookup(embeddings_matrix, A)
    B = tf.nn.embedding_lookup(embeddings_matrix, B)


    sampl_noise =tf.random_normal(shape=[int(A.shape[0]), 5,300], dtype = tf.float64)

    print("\n\n\n shapes", sampl_noise.shape, A.shape)

    A = tf.concat([A,sampl_noise], axis = 1)

    A = tf.reshape(A,(-1,19*300))
    B = tf.reshape(B,(-1,19*300))

    assert(A.shape == B.shape)
    A = A.eval(session=sess)
    B = B.eval(session =sess)

    N = int(B.shape[0])
    idxs = np.arange(N)
    if shuffle:
        np.random.shuffle(idxs)
    A_dev = A[num_train_examples:num_train_examples+num_dev_examples,:]
    A = A[:num_train_examples,:]

    B_dev = B[num_train_examples:num_train_examples+num_dev_examples,:]
    B = B[:num_train_examples,:]

    for epoch in range(num_iterations):
        for i in range(0, num_train_examples, batch_size): #these many iterations per epoch
            minibatch_b = B[i:i+batch_size]
            minibatch_a = A[i:i+batch_size]
            if minibatch_b.shape[0] != batch_size:
                continue
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={b: minibatch_b,a: minibatch_a})
            _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={a: minibatch_a})

        # print loss every so often.
        # We want to make sure D_loss doesn't go to 0
        if epoch % print_every == 0:
            ##print('epoch: {}, D: {:.4}, G:{:.4}'.format(epoch,D_loss_curr,G_loss_curr))
            dev_eval(A_dev, B_dev, batch_size, sess)
    samples = sess.run(G_sample)


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    session = tf.Session(config=config)
    return session



def lsgan_loss(score_real, score_fake):
    """Compute the Least Squares GAN loss.
    """
    
    true_labels = tf.ones_like(score_fake)
    fake_b_loss = tf.reduce_mean((score_real - true_labels)**2)
    real_b_loss = tf.reduce_mean(score_fake**2)
    D_loss = 0.5*(fake_b_loss + real_b_loss)
    
    G_loss = 0.5*tf.reduce_mean((score_fake - true_labels)**2)
    return D_loss, G_loss



###############################

tf.reset_default_graph()

batch_size = 128


# placeholder for input from the training dataset
b = tf.placeholder(tf.float32, [None, 19*300]) 

a = tf.placeholder(tf.float32, [None, 19*300]) 



G_sample = generator(a)
with tf.variable_scope("") as scope:
    
    #scale input to be -1 to 1
    logits_real = discriminator(b)
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    logits_fake = discriminator(G_sample)

# Get the list of variables for the discriminator and generator
D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator') 

# get our solver
D_solver, G_solver = get_solvers()

# get our loss
D_loss, G_loss = gan_loss(logits_real, logits_fake)

# setup training steps
D_train_step = D_solver.minimize(D_loss, var_list=D_vars)
G_train_step = G_solver.minimize(G_loss, var_list=G_vars)
D_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')
G_extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')




with get_session() as sess:
    sess.run(tf.global_variables_initializer())

    run_a_gan(sess,G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step)


