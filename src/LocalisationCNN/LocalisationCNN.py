"""
Localisation CNN
Author: Edward Ferdian
Date:   01/06/2018
"""
import tensorflow as tf
import numpy as np
import time
import datetime
import utils as log
import tf_util

class LocalisationCNN:
    # constructor
    def __init__(self, initial_learning_rate=1e-3, training_keep_prob=0.8):
        self.network_name = 'LocalCNN'
        
        self.output_size = 4 # The output is 2 bbox corner points (x1, y1, x2, y2)
        self.img_size = 256
        self.early_stop_threshold = 15
        self.training_keep_prob = training_keep_prob

        self.sess = tf.Session()
        
        # Placeholders & Vars
        self.x = tf.placeholder(tf.float32, [None, self.img_size, self.img_size], name='x1')
        self.y = tf.placeholder(tf.float32, [None, self.output_size], name='y')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')

        # Add an extra dimension (channel)
        x_shaped = tf.expand_dims(self.x, 3)
        
        # -- Main network --
        self.y_ = self.build_network(x_shaped, self.output_size, is_training=True, keep_prob=training_keep_prob)
        # Name output layer as tensor Variable so we can restore it easily
        self.y_ = tf.identity(self.y_, name="y_")

        # Loss, and acc
        self.iou = self.bbox_iou_corner_xy(bboxes1=self.y, bboxes2=self.y_)
        self.iou = tf.identity(self.iou, name="iou")

        self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.y_)
        self.loss = tf.identity(self.loss, name="loss")

        # learning rate and training optimizer
        self.learning_rate = tf.Variable( initial_value = initial_learning_rate, trainable = False, name = 'learning_rate' ) 
        self.adjust_learning_rate = tf.assign(self.learning_rate, self.learning_rate / np.sqrt( 2 ) )
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')

        # !!! By default the update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be added as a dependency to the train_op
        # refer to: https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss, name='train_op')

        # initialise the variables
        print("Initializing session...")
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver = tf.train.Saver()

    def init_model_dir(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        self.unique_model_name = '{}_{}'.format(self.network_name, timestamp)

        model_dir = "../models/{}".format(self.unique_model_name)
        # Do not use .ckpt on the model_path
        self.model_path = "{}/{}".format(model_dir, self.network_name)

        # summary - Tensorboard stuff
        self.create_summary('learning_rate', self.learning_rate)
        self.merged = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(model_dir+'/tensorboard/train', self.sess.graph)
        self.val_writer   = tf.summary.FileWriter(model_dir+'/tensorboard/validation')

    def restore_model(self, model_dir, model_name):
        print('Restoring model {}'.format(model_name))
        #new_saver = tf.train.import_meta_graph('{}/{}.meta'.format(model_dir, model_name))
        # Because we already have the graph, no need to import the meta graph anymore
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
        
        # Check memory usage for the loaded model
        tf_util.print_memory_usage(self.sess)

    def create_summary(self, tagname, value):
        """
            Create a scalar summary with the specified tagname
        """
        tf.summary.scalar('{}/{}'.format(self.network_name, tagname), value)

    def build_network(self, input_x, output_size, is_training, keep_prob=0.8):
        """
            Localisation CNN:
            Input: Image 256 x 256
            Output: Bounding box coordinates (x1, y1, x2, y2)
        """
        # Layer 1
        with tf.variable_scope("layer1"):
            conv1 = tf.layers.conv2d(input_x, filters=32, kernel_size=3, strides=1, padding='valid', activation=None, name='conv')
            conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            conv1 = tf.nn.relu(conv1)
            pool1 = tf.layers.max_pooling2d(conv1, 3, 2, 'valid', name='pool')
        
        # Layer 2
        with tf.variable_scope("layer2"):
            conv2 = tf.layers.conv2d(pool1, filters=128, kernel_size=3, strides=1, padding='valid', activation=None, name='conv')
            conv2 = tf.layers.batch_normalization(conv2, training=is_training)
            conv2 = tf.nn.relu(conv2)
            pool2 = tf.layers.max_pooling2d(conv2, 3, 2, 'valid', name='pool')

        # Layer 3
        with tf.variable_scope("layer3"):
            conv3 = tf.layers.conv2d(pool2, filters=128, kernel_size=3, strides=1, padding='valid', activation=None, name='conv')
            conv3 = tf.layers.batch_normalization(conv3, training=is_training)
            conv3 = tf.nn.relu(conv3)
            pool3 = tf.layers.max_pooling2d(conv3, 3, 2, 'valid', name='pool')
        
        # Layer 4
        with tf.variable_scope("layer4"):
            conv4 = tf.layers.conv2d(pool3, filters=128, kernel_size=3, strides=1, padding='valid', activation=None, name='conv')
            conv4 = tf.layers.batch_normalization(conv4, training=is_training)
            conv4 = tf.nn.relu(conv4)
            pool4 = tf.layers.max_pooling2d(conv4, 3, 2, 'valid', name='pool')
        
        # Layer 5
        with tf.variable_scope("layer5"):
            conv5 = tf.layers.conv2d(pool4, filters=128, kernel_size=3, strides=1, padding='valid', activation=None, name='conv')
            conv5 = tf.layers.batch_normalization(conv5, training=is_training)
            conv5 = tf.nn.relu(conv5)
            pool5 = tf.layers.max_pooling2d(conv5, 3, 2, 'valid', name='pool')
        
        # print("pool 5 shape", pool5.shape)
        flattened = tf.reshape(pool5, [-1, pool5.shape[1] * pool5.shape[2] * 128])

        with tf.variable_scope("dense1"):
            dense1 = tf.layers.dense(flattened, 1024, activation=tf.nn.relu, use_bias=True)
            dense1 = tf.layers.dropout(dense1, rate=1-keep_prob)

        with tf.variable_scope("dense2"):
            # Regression layer, no activation
            dense2 = tf.layers.dense(dense1, output_size, activation=None, use_bias=True)

        return dense2

    def bbox_iou_corner_xy(self, bboxes1, bboxes2):
        """
        Calculate Accuracy (IoU)

        Args:
            bboxes1: shape (total_bboxes1, 4)
                with x1, y1, x2, y2 point order.
            bboxes2: shape (total_bboxes2, 4)
                with x1, y1, x2, y2 point order.

            p1 *-----
            |     |
            |_____* p2

        Returns:
            Tensor with shape (total_bboxes1, total_bboxes2)
            with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
            in [i, j].
        """
        epsilon = 0.0001 # to avoid division by zero

        x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)

        xI1 = tf.maximum(x11, x21)
        xI2 = tf.minimum(x12, x22)

        yI1 = tf.maximum(y11, y21)
        yI2 = tf.minimum(y12, y22)

        inter_area = (xI1 - xI2) * (yI1 -yI2)

        bboxes1_area = (x12 - x11) * (y12 - y11)
        bboxes2_area = (x22 - x21) * (y22 - y21)

        union = (bboxes1_area + bboxes2_area) - inter_area

        iou = (inter_area+epsilon) / (union+epsilon)
        # print('iou',iou)
        return tf.reduce_mean(iou)

    def run_epoch(self, epoch_idx, next_element, is_training=False):
        """
            Run a single epoch (for validation or training).
            Running an epoch means looping through the whole batch iterator.
            The batch will be exhausted once this run is finished. 
            Note: iterator must be initialized first outside of this function.
        """
        start_loop = time.time()

        total_loss = 0
        total_acc = 0
        
        total_data = 0
        try:
            i = 0
            while True:
                # print(total_data)
                # Get input and label batch
                next_batch = self.sess.run(next_element)
                batch_x = next_batch[0] # Image
                batch_y = next_batch[1] # Label
                
                n_per_batch = len(batch_x)

                if is_training:
                    # Feed the network and optimize
                    feed = {self.x: batch_x, self.y: batch_y, self.is_training: True, self.keep_prob: self.training_keep_prob}
                    _, merged_summ, cost, acc = self.sess.run([self.train_op, self.merged, self.loss, self.iou], feed_dict=feed)
                else:
                    # Feed it to the network
                    feed = {self.x: batch_x, self.y: batch_y, self.is_training: False, self.keep_prob: 1}
                    cost, acc = self.sess.run([self.loss, self.iou], feed_dict=feed)
                
                total_loss += cost * n_per_batch
                total_acc  += acc * n_per_batch
                total_data += n_per_batch

                print ("\rRead %d rows: [%-30s], batch loss %.3f | Elapsed: %.2f secs." % (total_data, '='*(i//5), cost, time.time()-start_loop), end='')
                i += 1

        except tf.errors.OutOfRangeError:
            # Without .repeat(), iterator is exhaustive. This is a common practice
            # If we want to use repeat, then we need to specify the number of batch, instead of using 'while' loop
            print ("\rRead %d rows: [%-30s], batch loss %.3f | Elapsed: %.2f secs." % (total_data, '='*30, cost, time.time()-start_loop), end='')
            pass

        # calculate the avg loss per epoch
        avg_cost = total_loss / total_data   
        avg_acc  = total_acc / total_data
            
        end_loop = time.time()

        # Log and summary
        summary = tf.Summary()
        summary.value.add(tag='{}/Avg_loss'.format(self.unique_model_name), simple_value=avg_cost)
        summary.value.add(tag='{}/Accuracy'.format(self.unique_model_name), simple_value=avg_acc)

        if is_training:
            epoch_name = "Training"
            self.train_writer.add_summary(summary, epoch_idx)
            self.train_writer.add_summary(merged_summ, epoch_idx) # standard merged summ
        else:
            epoch_name = "Validation"
            self.val_writer.add_summary(summary, epoch_idx)

        msg = "\n{} {}\t- Loss (MSE): {:.3f}, IoU: {:.3f}, time elapsed  : {:.2f} seconds".format(epoch_idx+1, epoch_name, avg_cost, avg_acc, end_loop-start_loop)
        log.info(msg)
        
        return avg_cost, avg_acc
        
    def run_prediction_epoch(self, next_element):
        """
            Run a single epoch (for prediction).
            Note: iterator must be initialized first outside of this function.
        """
        start_loop = time.time()

        total_loss = 0
        total_acc = 0
        
        total_data = 0
        predictions = np.empty((0,4)) 

        try:
            while True:
                # print(total_data)
                # Get input and label batch
                next_batch = self.sess.run(next_element)
                batch_x = next_batch[0] # Image
                batch_y = next_batch[1] # Label
                
                n_per_batch = len(batch_x)

            
                # Feed it to the network
                feed = {self.x: batch_x, self.y: batch_y, self.is_training: False, self.keep_prob: 1}
                cost, acc, prediction = self.sess.run([self.loss, self.iou, self.y_], feed_dict=feed)
                
                prediction = np.asarray(prediction)
                # print(prediction.shape)

                predictions = np.append(predictions, prediction, axis=0)
                total_loss += cost * n_per_batch
                total_acc  += acc * n_per_batch
                total_data += n_per_batch

                #print (total_loss, total_acc)
        except tf.errors.OutOfRangeError:
            # Without .repeat(), iterator is exhaustive. This is a common practice
            # If we want to use repeat, then we need to specify the number of batch, instead of using 'while' loop
            pass

        # calculate the avg loss per epoch
        avg_cost = total_loss / total_data   
        avg_acc  = total_acc / total_data
            
        end_loop = time.time()

        # Log 
        msg = "\n{}\t- Loss (MSE): {:.3f}, IoU: {:.3f}, time elapsed  : {:.2f} seconds".format("Test", avg_cost, avg_acc, end_loop-start_loop)
        log.info(msg)
        return predictions, avg_cost, avg_acc

    def predict(self, data_iterator):
        next_element = data_iterator.get_next()
        self.sess.run(data_iterator.initializer)

        predictions, cost, accuracy = self.run_prediction_epoch(next_element)
        print(np.asarray(predictions).shape)


    def train_network(self, train_iterator, val_iterator, n_epoch):
        log.info("==================== TRAINING =================")
        log.info("Starting the training for localisation network at {}\n".format(time.ctime()))
        start_time = time.time()

        # setting up
        next_element = train_iterator.get_next()
        next_validation = val_iterator.get_next()
        
        previous_acc = 0
        last_saved_epoch = 0
        for epoch in range(n_epoch):
            # reinitialize iterator every epoch to reset it back
            self.sess.run(train_iterator.initializer)
            self.sess.run(val_iterator.initializer)

            log.info("\nEpoch {} {}".format((epoch+1), time.ctime()))
            
            # Reduce learning rate every few steps
            if epoch >= 10 and epoch % 5 == 0:
                self.adjust_learning_rate.eval(session=self.sess)
                log.info('Learning rate adjusted to {}'.format(self.sess.run(self.learning_rate)))

            # Train on all batches in training set
            self.run_epoch(epoch, next_element, is_training=True)
            
            # Validate on all batches in validation set
            val_cost, validation_acc = self.run_epoch(epoch, next_validation, is_training=False)

            # ------------------------------- Save the weights -------------------------------
            if validation_acc > previous_acc:
                # Save model weights to disk whenever the validation acc reaches a new high
                save_path = self.saver.save(self.sess, self.model_path)
                log.info("Model saved in file: %s" % save_path)
                
                # Update the cost for saving purposes
                last_saved_epoch = epoch
                previous_acc = validation_acc
            else:
                # Early stopping
                if last_saved_epoch > 0: # only do this if we ever saved before
                    if ((epoch - last_saved_epoch) >= self.early_stop_threshold):
                        msg = '\nEpoch {} - Early stopping, no more validation acc increase after {} epochs'.format(epoch+1, self.early_stop_threshold)
                        log.info(msg)
                        break
        # /END of epoch loop

        log.info("\nTraining for localisation CNN completed!")
        hrs, mins, secs = log.calculate_time_elapsed(start_time)
        log.info("Total training time: {} hrs {} mins {} secs.".format(hrs, mins, secs))
        log.info("Finished at {}".format(time.ctime()))
        log.info("==================== END TRAINING =================")

