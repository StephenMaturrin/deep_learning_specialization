import tensorflow as tf
from keras import activations
from tensorflow.contrib.layers.python.layers.initializers import xavier_initializer
from  tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "./log"
logdir = "{}/run-{}/".format(root_logdir, now)


mnist = input_data.read_data_sets("./data/data")


n_hidden_L1 = 28*28
n_hidden_L2 = 28*28
n_output = 10

def train():
    X = tf.placeholder(tf.float32,shape=(None,n_hidden_L1), name="X" )
    Y = tf.placeholder(tf.int64,shape=(None), name ="Y")


    # with tf.name_scope("layer-1"):
    #     W1 = tf.get_variable("W1", shape=(n_hidden_L1,n_hidden_L2),initializer=tf.contrib.layers.xavier_initializer())
    #     B1 = tf.Variable(tf.zeros([n_hidden_L2]))
    #     layer_1 = tf.add(tf.matmul(X,W1),B1)
    #     layer_1_act = tf.nn.selu(layer_1)
    #
    #     tf.summary.histogram(layer_1, name="layer1")
    #     tf.summary.scalar(tf.reduce_mean(layer_1_act))
    #     tf.summary.histogram(W1,name="W1")
    #     tf.summary.histogram(B1, name= "B1")


    def variable_summaries(var,name):
        with tf.name_scope("summaries/"+name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar("mean_"+name, mean)
            with tf.name_scope("stddev"):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
                tf.summary.scalar("stddev_"+name,stddev)
                tf.summary.scalar("max_"+name, tf.reduce_max(var))
                tf.summary.scalar("min_"+name, tf.reduce_min(var))
                tf.summary.histogram("h_"+name,var)

    def nn_layer(input_tensor, input_dim, output_dim, layer_name, activation_fn= tf.nn.selu):

        with tf.name_scope(layer_name):
            with tf.name_scope("Weights"):
                weights = tf.get_variable("Weight_"+layer_name,shape=[input_dim,output_dim], initializer= tf.contrib.layers.xavier_initializer())
                variable_summaries(weights,layer_name)
            with tf.name_scope("Biases"):
                bias = tf.Variable(tf.zeros(shape=output_dim), "Bias_"+layer_name)
                variable_summaries(bias,layer_name)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.add(tf.matmul(input_tensor,weights),bias)
                tf.summary.histogram('pre_activation_'+layer_name, preactivate)
            activation = activation_fn(preactivate)
            tf.summary.histogram('activation_'+layer_name, activation)
            return activation


    hidden1 = nn_layer(X,n_hidden_L1,500,'layer')
    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar("dropout_keep_probability", keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)


    hypothesis = nn_layer(dropped,500,10,'layer2', activation_fn=tf.identity)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=hypothesis)

    tf.summary.scalar("cross_entropy", cross_entropy)

    with tf.name_scope("train"):
        train_step= tf.train.AdamOptimizer(learning_rate= 0.001).minimize( cross_entropy)


    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.arg_max(hypothesis,1), Y)
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)


    merged = tf.summary.merge_all()

    sess = tf.InteractiveSession()

    train_writer = tf.summary.FileWriter(logdir+'/train',sess.graph)
    test_writer = tf.summary.FileWriter(logdir+'/test',sess.graph)
    tf.global_variables_initializer().run()



    def feed_dict(train):
        if train:
            xs , ys = mnist.train.next_batch(100)
            k = 0.5
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1
        return {X: xs, Y : ys, keep_prob : k }

    steps = 1000
    for i in range(steps):
        if i % 10 ==0:
            summary, acc = sess.run([merged,accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary,i)
        else:
            if i % 100 == 99:
                run_options= tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict= feed_dict(True),
                                      options= run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step_%04d' % i)
                train_writer.add_summary(summary,i)
            else:
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary,i)
    train_writer.close()
    test_writer.close()


train()

print("tensorboard --logdir=%s --port=" %logdir)






