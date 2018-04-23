import tensorflow as tf
import time
import numpy as np
import expression_generator
import os

epochs = 10000
batch_size = 128
learning_rate = 1e-3

tf.reset_default_graph()
sess = tf.InteractiveSession()

nodes = 128
embed_size = 20
vocab_size = expression_generator.vocab_size

input_length = 4
output_length = 3

inputs = tf.placeholder(tf.int32, (None, input_length), name = 'inputs')
outputs = tf.placeholder(tf.int32, (None, None), name = 'output')
targets = tf.placeholder(tf.int32, (None, None), name = 'targets')

with tf.variable_scope("embeding"):
    input_embedding = tf.Variable(tf.random_uniform((vocab_size, embed_size), -1.0, 1.0), name='encoder_embedding')
    output_embedding = tf.Variable(tf.random_uniform((vocab_size, embed_size), -1.0, 1.0), name='decoder_embedding')
    input_embed = tf.nn.embedding_lookup(input_embedding, inputs)
    output_embed = tf.nn.embedding_lookup(output_embedding, outputs)

with tf.variable_scope("encoder"):
    encoder = tf.contrib.rnn.BasicLSTMCell(nodes)
    _, last_state = tf.nn.dynamic_rnn(encoder, inputs=input_embed, dtype=tf.float32)

with tf.variable_scope("decoder"):
    decoder = tf.contrib.rnn.LSTMCell(nodes)
    dec_outputs, _ = tf.nn.dynamic_rnn(decoder, inputs=output_embed, initial_state=last_state)
    logits = tf.contrib.layers.fully_connected(dec_outputs, num_outputs=vocab_size, activation_fn=None)

with tf.variable_scope("loss"):
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, output_length]))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)


saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter("output", sess.graph)

step = epochs
checkpoint = tf.train.get_checkpoint_state("model")
if checkpoint and checkpoint.model_checkpoint_path:
    s = saver.restore(sess,checkpoint.model_checkpoint_path)
    print("Loaded model:", checkpoint.model_checkpoint_path)
    step = int(os.path.basename(checkpoint.model_checkpoint_path).split('-')[1])
else:
    print("Can't find model")


    try:
        start_time = time.time()
        for i in range(epochs+1):
            epoch_start_time = time.time()
            batch_x, batch_y = expression_generator.get_data(batch_size)
            _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
                    feed_dict ={
                        inputs: batch_x,
                        outputs: [[expression_generator.EOS] + sequence for sequence in batch_y],
                        targets: [sequence + [expression_generator.EOS] for sequence in batch_y]
                    })

            if i % 100 == 0:
                print('Epoch: ' + str(i) + ", Loss: " + str(batch_loss) + ", Epoch duration: " + str(time.time() - epoch_start_time) + ", Total time: " + str(time.time() - start_time))
                saver.save(sess, "./model/model.ckpt", global_step=i)
    except KeyboardInterrupt:
        print('training interrupted')

#TEST

NUM_TEST = 1000
success = 0
fail = 0

for _ in range(NUM_TEST):
    x, y = expression_generator.get_data(1)
    decoder_input = np.zeros((len(x), 1)) + expression_generator.START_TOKEN
    result = []
    for i in range(3):
        batch_logits = sess.run(logits,
                                feed_dict={inputs: x,
                                           outputs: decoder_input})
        prediction = batch_logits[:, -1].argmax(axis=-1)
        result.append(prediction.tolist())
        decoder_input = np.hstack([decoder_input, prediction[:, None]])

    y = expression_generator.array2expression(y)
    result = expression_generator.array2expression(result)
    try:
        if int(result) == int(y):
            success += 1
        else:
            fail += 1
            print("_______________")
            print("X")
            print(expression_generator.array2expression(x))
            print("Y")
            print(y)
            print("Pred")

            print(result)
    except:
        fail += 1
        print(result)

print(str(success/NUM_TEST * 100) + "% success rate")

#VISUAL TESTS
for _ in range(10):
    print("_______________")

    x, y = expression_generator.get_data(1)

    print(x)

    print("X")
    print(expression_generator.array2expression(x))

    decoder_input = np.zeros((len(x), 1)) + expression_generator.START_TOKEN
    result = []
    for i in range(3):
        batch_logits = sess.run(logits,
                                feed_dict={inputs: x,
                                           outputs: decoder_input
                                           })
        prediction = batch_logits[:, -1].argmax(axis=-1)
        result.append(prediction.tolist())
        print("Prediction: " + str(prediction))
        decoder_input = np.hstack([decoder_input, prediction[:, None]])

    y = expression_generator.array2expression(y)
    result = expression_generator.array2expression(result)


    print("Y")
    print(y)
    print("Pred")

    print(result)

writer.close()

