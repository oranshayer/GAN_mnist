import tensorflow as tf

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
GEN_INPUT = 100

LEARNING_RATE = 0.001
BATCH_SIZE = 128

def inference_generative(Z):
    vars = []
    # Hidden layer    
    with tf.variable_scope('hidden_G') as scope:
        weights = tf.get_variable('weights',[GEN_INPUT,128],initializer=tf.contrib.layers.xavier_initializer(uniform=False),dtype=tf.float32)
        biases = tf.get_variable('biases',[128],initializer=tf.constant_initializer(0.0),dtype=tf.float32)
        hidden_G = tf.nn.relu(tf.matmul(Z, weights) + biases, name=scope.name)
        vars = vars + [weights, biases]

    # Out layer    
    with tf.variable_scope('out_G') as scope:
        weights = tf.get_variable('weights',[128,IMAGE_PIXELS],initializer=tf.contrib.layers.xavier_initializer(uniform=False),dtype=tf.float32)
        biases = tf.get_variable('biases',[IMAGE_PIXELS],initializer=tf.constant_initializer(0.0),dtype=tf.float32)
        out_G = tf.nn.sigmoid(tf.matmul(hidden_G, weights) + biases, name=scope.name)
        vars = vars + [weights, biases]
        
    return out_G, vars

def inference_descriminator(X, is_real):
    vars = []
    if is_real:
        X_reshaped = tf.reshape(X, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
        tf.image_summary('mnist', X_reshaped, max_images=6)
    # Hidden layer
    with tf.variable_scope('hidden_D') as scope:
        weights = tf.get_variable('weights',[IMAGE_PIXELS,128],initializer=tf.contrib.layers.xavier_initializer(uniform=False),dtype=tf.float32)
        biases = tf.get_variable('biases',[128],initializer=tf.constant_initializer(0.0),dtype=tf.float32)
        hidden_D = tf.nn.relu(tf.matmul(X, weights) + biases, name=scope.name)
        vars = vars + [weights, biases]

    # Out layer
    with tf.variable_scope('out_D') as scope:
        weights = tf.get_variable('weights',[128,1],initializer=tf.contrib.layers.xavier_initializer(uniform=False),dtype=tf.float32)
        biases = tf.get_variable('biases',[1],initializer=tf.constant_initializer(0.0),dtype=tf.float32)
        out_D = tf.nn.sigmoid(tf.matmul(hidden_D, weights) + biases, name=scope.name)
        vars = vars + [weights, biases]

    return out_D, vars
    
def loss_generator(probabilities):
    loss = -tf.reduce_mean(tf.log(probabilities))
    tf.scalar_summary('Generative loss', loss)
    return loss
    
def loss_discriminator(probs_real, probs_fake):
    loss = -tf.reduce_mean(tf.log(probs_real) + tf.log(1. - probs_fake))
    tf.scalar_summary('Discriminator loss', loss)
    return loss

def train(G_loss, D_loss, G_vars, D_vars):
#    D_train_op = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9).minimize(D_loss, var_list=D_vars)
#    G_train_op = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9).minimize(G_loss, var_list=G_vars)
#    
    
#    for var in G_vars:
#        print(var.name)
#    for var in D_vars:
#        print(var.name)
    
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
#    return D_train_op, G_train_op
    D_opt = tf.train.AdamOptimizer()#MomentumOptimizer(LEARNING_RATE, 0.9)
    G_opt = tf.train.AdamOptimizer()#MomentumOptimizer(LEARNING_RATE, 0.9)
    D_grads = D_opt.compute_gradients(D_loss, var_list=D_vars)
    G_grads = G_opt.compute_gradients(G_loss, var_list=G_vars)
    D_apply_gradient_op = D_opt.apply_gradients(D_grads)
    G_apply_gradient_op = G_opt.apply_gradients(G_grads)
    
    for grad, var in D_grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    for grad, var in G_grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    return D_apply_gradient_op,G_apply_gradient_op
