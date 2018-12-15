import math, random
from collections import defaultdict
import tensorflow as tf
import numpy as np
import pickle
from comet_ml import Experiment


experiment = Experiment(api_key="")

TF_CPP_MIN_LOG_LEVEL=2


def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)



def generator(z, alpha=0.2):
    """Generate b' conditional on input a = z
    """
    with tf.variable_scope("generator"):        

        #fc1 = tf.layers.dense(z, 30*10*256)

        # Reshape it to start the convolutional stack
        vocab_size =300

        conv1 = tf.layers.Conv1D(filters = 32,kernel_size = 1,strides = 1, padding='SAME')(z)
        conv1 = tf.layers.batch_normalization(conv1)#, training=training)
        conv1 = lrelu(conv1, alpha)

        print("\n\n\n shape of conv1 discr", conv1.shape)
        conv2 = tf.layers.Conv1D(filters = 64,  padding='SAME', strides = 1, kernel_size = 1)(conv1)
        conv2 = tf.layers.batch_normalization(conv2)#, training=training)
        conv2 = lrelu(conv2, alpha)

        conv3= tf.layers.Conv1D(filters=vocab_size,padding='SAME',strides = 1, kernel_size = 1)(conv2)
        conv3 = tf.layers.batch_normalization(conv3)#, training=training)
        logits = lrelu(conv3, alpha)

        #out = tf.nn.softmax(logits)

        gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
        out =  tf.nn.softmax(gumbel_softmax_sample) # by temp
        print("\n\n\nshape of output of generator", out.shape)
        return out

def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=0.01)
    return -tf.log(-tf.log(U + eps) + eps)

## softmax
##  19, vocab 


def discriminator(x,reuse=False, alpha=0.2, training=True):
    """Compute discriminator score for a batch of input bs.

    """
    vocab_size = 300
    print("\n\n, discriminator input", x.shape)
    with tf.variable_scope("discriminator"):
        conv1 = tf.layers.Conv1D(filters = 32,kernel_size = 1,strides = 1, padding='SAME')(x)

        conv1 = lrelu(conv1, alpha)
        print("\n\n\n shape of conv1 discr", conv1.shape)
        conv2 = tf.layers.Conv1D(filters = 64,  padding='SAME', strides = 1, kernel_size = 1)(conv1)
        conv2 = tf.layers.batch_normalization(conv2)#, training=training)
        conv2 = lrelu(conv2, alpha)

        conv3= tf.layers.Conv1D(filters=64,padding='SAME',strides = 1, kernel_size = 1)(conv2)
        conv3 = tf.layers.batch_normalization(conv3)#, training=training)
        conv3 = lrelu(conv3, alpha)

        print("\n\n\nshape of conv3", conv3.shape)
        # Flatten it
        flat = tf.reshape(conv3, (-1, 19*64))
        logits = tf.layers.dense(flat, 1)

        out = tf.sigmoid(logits)
        return  logits


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

def w_gan_loss(logits_real, logits_fake):
   
    D_loss = tf.reduce_mean(logits_fake - logits_real)

    G_loss = -tf.reduce_mean(logits_fake) 
    return D_loss, G_loss

# create an AdamOptimizer for D_solver and G_solver
def get_solvers(learning_rate=1e-3, beta1=0.5):
    """Create solvers for GAN training.
    """
    
    D_solver = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
    
    return D_solver, G_solver


def w_get_solvers(learning_rate=1e-3, beta1=0.5):
    """Create solvers for GAN training.
    """
    
    #D_solver = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
    D_solver = tf.train.RMSPropOptimizer(learning_rate=0.00005)
    G_solver = tf.train.RMSPropOptimizer(learning_rate=0.00005)
    
    return D_solver, G_solver

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
    return epoch_D_loss / examples, epoch_G_loss / examples 

def run_inference(A, B, B_fake_probs, dictionary):
    print("\n\n\n running inference")
    print("type", type(A[0][0]), type(A))
    B_fake_probs = np.argmax(B_fake_probs, axis = -1)
    print("shape of B and B'", B.shape , B_fake_probs.shape)
    for i in range(3):
        print("A:")
        print(ids_to_text(A[i], dictionary))
        print("B:")
        print(ids_to_text(B[i], dictionary))
        print("B':")
        print(ids_to_text(B_fake_probs[i], dictionary))


def ids_to_text(ids, dictionary):
    return " ".join([dictionary.reverse(ids[i]) for i in range(len(ids))])


def clipping_ops():
    clip_ops = []
    for var in tf.global_variables():
        if "discriminator" in var.name and "Adam" not in var.name:
            clip_ops.append(
                tf.assign(
                    var,
                    tf.clip_by_value(var, -0.4, 0.4)
                )
            )
    clip_disc_weights = tf.group(*clip_ops)
    return clip_disc_weights

def run_a_gan(sess, G_train_step, G_loss, D_train_step, D_loss, G_extra_step, D_extra_step,\
              G_sample,  print_every=5, batch_size=128,shuffle = False,\
              num_epochs = 90000000):
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

    num_dev_examples = batch_size*10
    vocab_size = 300
    num_train_examples = 10000

    # ## testing
    # A = np.random.rand(128*2,19, 300) #A[:128*2]
    # A = A.astype(int)

    # B = np.random.rand(128*2,19) 
    # B = B.astype(int)



    embeddings_matrix = np.load('../../data/wm.npy')


    labels = np.load('../../data/labels.npy')

    A = np.load('../../data/A_14.npy') #shape [none, 14]
    A = A.astype(int)
    A = A[~(labels == 0)]
    A = A[:num_train_examples + num_dev_examples,:]

    print("\n\n shapeof A", A.shape)


    B = np.load('../../data/B_19.npy', encoding = 'latin1')
    B = B.astype(int)
    B = B[~(labels== 0)]
    B = B[:num_train_examples + num_dev_examples,:]

    dictionary = pickle.load(open('../../data/vocabproc_maxl_20_minl_5_year_2016.pickle', 'rb'))

    A_dev_id = A[num_train_examples:num_train_examples + num_dev_examples,:]
    B_dev_id = B[num_train_examples:num_train_examples + num_dev_examples,:]
    print("\dev ids", A_dev_id.shape, B_dev_id.shape)
    A = tf.nn.embedding_lookup(embeddings_matrix, A)
    # B = tf.nn.embedding_lookup(embeddings_matrix, B)

    sampl_noise =tf.random_normal(shape=[int(A.shape[0]), 5,300], dtype = tf.float64)

    A = tf.concat([A,sampl_noise], axis = 1)

    print("shape of a",a.shape)
    
    B = tf.one_hot(indices=B, depth=vocab_size, dtype = tf.float32)

    print("\n\n\n shape of B", B.shape)
    A = A.eval(session=sess)
    B = B.eval(session =sess)

    A_dev = A[num_train_examples:num_train_examples+num_dev_examples,:]
    A = A[:num_train_examples,:]

    B_dev = B[num_train_examples:num_train_examples+num_dev_examples,:]
    B = B[:num_train_examples,:]
    # right now, their shape is none, 19, 300

    N = int(B.shape[0])
    idxs = np.arange(N)
    if shuffle:
        np.random.shuffle(idxs)

    dev_D_loss, dev_G_loss = 0,0 
    for epoch in range(num_epochs):
        for i in range(0, N, batch_size): #these iterations per epoch
            minibatch_b = B[i:i+batch_size]
            minibatch_a = A[i:i+batch_size]
            if minibatch_b.shape[0] != batch_size:
                continue
            probs_ori =sess.run(G_sample)
            new_b = sess.run(get_best_fake(G_sample))
            _, D_loss_curr = sess.run([D_train_step, D_loss], feed_dict={b: minibatch_b,a: minibatch_a,\
                    new_b : new_b})
            if (i%5 ==1) :
                _, G_loss_curr = sess.run([G_train_step, G_loss], feed_dict={a: minibatch_a})
            
            # print loss every so often.
            # We want to make sure D_loss doesn't go to 0
        #sess.run(clipping_ops())
        print( "\n\n epoch", epoch)
        if epoch % print_every == 0:
            ##print('epoch: {}, D: {:.4}, G:{:.4}'.format(epoch,D_loss_curr,G_loss_curr))
            print("\n\nepoch", epoch)
            dev_D_loss, dev_G_loss =dev_eval(A_dev, B_dev, batch_size, sess)

            generated_probs = sess.run([G_sample], feed_dict={a: A_dev})
            print("softmax probabilities' shape", generated_probs[0].shape, type(generated_probs[0]))
            run_inference(A_dev_id, B_dev_id, generated_probs[0], dictionary)

    # samples = sess.run(G_sample)

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


def get_best_fake(G_sample):
    with get_session() as sess:
        sess.run(tf.global_variables_initializer())
        #B_ind = np.argmax(G_sample, axis = -1)
        B_ind = tf.argmax(G_sample, axis=-1)
        print("\n\n\n shape off b argmax", B_ind.shape, type(B_ind) )
        B_dash_onehot = tf.one_hot(indices=B_ind, depth=300, dtype = tf.float32)
        print(B_dash_onehot[:,:,1])
        return B_dash_onehot
##########################

tf.reset_default_graph()

vocab_size = 300
b = tf.placeholder(tf.float32, [None,19, vocab_size]) #
new_b = tf.placeholder(tf.float32, [None,19, vocab_size]) #


a = tf.placeholder(tf.float32, [None, 19, vocab_size]) 



# generated input
G_sample = generator(a)
print("generator output" ,G_sample.shape, type(G_sample), "\n\n\n")
#b = tf.expand_dims(b, 1)
#G_sample  = tf.expand_dims(G_sample, 1)
with tf.variable_scope("") as scope:
    #scale input to be -1 to 1

    logits_real = discriminator(b)
    print("shape of logits", logits_real.shape)
    # Re-use discriminator weights on new inputs
    scope.reuse_variables()
    #logits_fake = discriminator(get_best_fake(G_sample))
    #logits_fake = discriminator(G_sample)
    logits_fake = discriminator(new_b)

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
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    sess.run(tf.global_variables_initializer(), options=run_options)
    run_a_gan(sess,G_train_step,G_loss,D_train_step,D_loss,G_extra_step,D_extra_step,G_sample)

