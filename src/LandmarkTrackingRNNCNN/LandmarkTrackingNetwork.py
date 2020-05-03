"""
Landmark Tracking RNNCNN Network
Author: Edward Ferdian
Date:   01/06/2018
"""
import tensorflow as tf
import numpy as np
import time
import datetime
import utils as log
import loss_util

def leaky_relu(x, alpha=0.1):
    return tf.nn.leaky_relu(x, alpha=alpha)

class LandmarkTrackingNetwork:
    # constructor
    def __init__(self, batch_size, initial_learning_rate=1e-4, training_keep_prob=0.8):
        self.network_name = 'LandmarkTrackingRNNCNN'
        
        self.trunc_time_steps = 20
        self.batch_size = batch_size

        # Input: 20 128x128 => 20 frames of 128x128 images (cropped around myocardium)
        # Output: 168 landmark coordinates
        self.output_size = 2 * 168 
        self.state_size = 1024
        self.batch_size = batch_size

        self.training_keep_prob = training_keep_prob

        self.sess = tf.Session()

        # --- Placeholders & Vars ---
        # x: (batch_size, time_steps, height, width)
        x = tf.placeholder(tf.float32, [None, None, 128, 128], name='x')
        self.x = tf.expand_dims(x,4) # Add an extra dimension (channel)
        
        # y: (batch_size, time_steps, 2*168)
        self.y = tf.placeholder(tf.float32, [None, None, self.output_size], name='y')
        
        self.init_state = tf.placeholder(tf.float32, [2, None, self.state_size], name='init_state')
        self.keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')

        # ----- Main network -----
        self.y_ = self.build_network(self.x)
        self.y_ = tf.identity(self.y_, name="y_")

        # print(self.y_.shape, "output shape")

        # Calculate loss
        log.info("Preparing loss function")
        self.loss2, self.rr_error, self.cc_error = loss_util.calculate_loss(labels=self.y, predictions=self.y_, time_steps=self.trunc_time_steps)
        
        # learning rate and training optimizer
        self.learning_rate = tf.Variable( initial_value = initial_learning_rate, trainable = False, name = 'learning_rate' ) 
        self.adjust_learning_rate = tf.assign(self.learning_rate, self.learning_rate / np.sqrt( 2 ) )
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss2, name='train_op')

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

    def create_summary(self, tagname, value):
        """
            Create a scalar summary with the specified tagname
        """
        tf.summary.scalar('{}/{}'.format(self.network_name, tagname), value)

    def restore_model(self, model_dir, model_name):
        print('Restoring model {}'.format(model_name))
        #new_saver = tf.train.import_meta_graph('{}/{}.meta'.format(model_dir, model_name))
        # Because we already have the graph, no need to import the meta graph anymore
        self.saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))


    def build_network(self, image_sequences):
        """
            Overall RNNCNN architecture
            Input: Image sequences (batch_size, time_steps, height, width)
            Output: 168 Landmarks per frame (batch size, time_steps, 2, 168)
        """        
        # ------------- CNN component -------------
        log.info("Building CNN cells")
        # extract spatial features for each time t up to the specified time steps
        flat_stack = []        
        for t in range(0, self.trunc_time_steps):    
            img = image_sequences[:,t,:,:, :]
            feature_vector = self.build_cnn_cell(img)
            flat_stack.append(feature_vector)
        
        # batches x time_steps x embedding
        cnn_outputs = tf.concat(flat_stack, axis=1)

        #------------- RNN units -------------
        log.info("Building RNN component")
        basic_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=1024)
        basic_cell = tf.nn.rnn_cell.DropoutWrapper(basic_cell, output_keep_prob=self.keep_prob)
        rnn_outputs, states = tf.nn.dynamic_rnn(cell=basic_cell, inputs=cnn_outputs, dtype=tf.float32, time_major=False)


        rnn_outputs = tf.identity(rnn_outputs, name="rnn_outputs")
        self.states = tf.identity(states, name="states")

        # ------------- fully connected layers -------------
        dense1 = tf.layers.dense(rnn_outputs, 1024, activation=tf.nn.relu, use_bias=True)
        dense1 = tf.layers.dropout(dense1, rate=1-self.keep_prob)

        # regression layer
        dense2 = tf.layers.dense(dense1,units=self.output_size,activation=None, use_bias=True)
        return dense2

    def build_cnn_cell(self, image):
        """
            CNN cell
            Spatial feature extraction, shared weight using tf.AUTO_REUSE 
        """
        with tf.variable_scope('cnn_cell', reuse = tf.AUTO_REUSE ):
            # Layer 1
            layer1 = tf.layers.conv2d(inputs=image,filters=32,kernel_size=3, strides=(1,1), padding="VALID", activation=leaky_relu ,name='conv1')
            pool1 = tf.layers.max_pooling2d(layer1, 2, 2, padding='VALID', name='pool1')
            
            # Layer 2
            layer2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=3, strides=(1,1), padding="VALID", activation=leaky_relu ,name='conv2')
            pool2 = tf.layers.max_pooling2d(layer2, 2, 2, padding='VALID', name='pool2')
            
            # Layer 3
            layer3 = tf.layers.conv2d(inputs=pool2,filters=64,kernel_size=3, strides=(1,1), padding="VALID", activation=leaky_relu ,name='conv4')
            layer3 = tf.layers.max_pooling2d(layer3, 2, 2, padding='VALID', name='pool3')

            # Layer 5
            layer5 = tf.layers.conv2d(inputs=layer3,filters=128,kernel_size=3, strides=(1,1), padding="SAME", activation=leaky_relu ,name='conv8')
            # transpose flatten just like YOLO
            dim = layer5.shape[1] * layer5.shape[2] * layer5.shape[3]
            layer5_transposed = tf.transpose(layer5,(0,3,1,2))
            flattened = tf.reshape(layer5_transposed, [-1, 1, dim])
            
            dense1 = tf.layers.dense(flattened, 1024, activation=leaky_relu, use_bias=True)

            return dense1

    def predict(self, data_iterator):
        next_element = data_iterator.get_next()
        self.sess.run(data_iterator.initializer)

        predictions, cost = self.run_prediction_epoch(next_element)
        print(np.asarray(predictions).shape)

    def train_network(self, train_iterator, val_iterator, n_epoch):
        log.info("==================== TRAINING =================")
        log.info("Starting the training for landmark tracking network at {}\n".format(time.ctime()))
        start_time = time.time()

        # setting up
        log.info('Setting init state to 0')
        _current_state = np.zeros((2, self.batch_size, self.state_size))

        next_element = train_iterator.get_next()
        next_validation = val_iterator.get_next()
        
        previous_low = 9999 # just a random number for initial cost

        for epoch in range(n_epoch):
            # reinitialize iterator every epoch to reset it back
            self.sess.run(train_iterator.initializer)
            self.sess.run(val_iterator.initializer)

            log.info("\nEpoch {} {}".format((epoch+1), time.ctime()))
            
            # Reduce learning rate every few steps
            if epoch >= 10 and epoch % 10 == 0:
                self.adjust_learning_rate.eval(session=self.sess)
                log.info('Learning rate adjusted to {}'.format(self.sess.run(self.learning_rate)))

            # Train on all batches in training set
            train_loss, _current_state = self.run_epoch(epoch, next_element, _current_state, is_training=True)
            
            # Validate on all batches in validation set
            val_loss, _current_state = self.run_epoch(epoch, next_validation, _current_state, is_training=False)

            # ------------------------------- Save the weights -------------------------------
            if val_loss < previous_low:
                # Save model weights to disk whenever the validation acc reaches a new high
                save_path = self.saver.save(self.sess, self.model_path)
                log.info("Model saved in file: %s" % save_path)
                
                # Update the cost for saving purposes
                previous_low = val_loss
        # /END of epoch loop

        log.info("\nTraining RNNCNN completed!")
        hrs, mins, secs = log.calculate_time_elapsed(start_time)
        log.info("Total training time: {} hrs {} mins {} secs.".format(hrs, mins, secs))
        log.info("Finished at {}".format(time.ctime()))
        log.info("==================== END TRAINING =================")

    def run_prediction_epoch(self, next_element):
        """
            Run a single epoch (for validation or training).
            Running an epoch means looping through the whole batch iterator.
            The batch will be exhausted once this run is finished. 
            Note: iterator must be initialized first outside of this function.
        """
        start_loop = time.time()
        total_loss = 0
        total_data = 0
        predictions = np.empty((0,self.trunc_time_steps, 2, 168))

        try:
            i = 0
            while True:
                # Get input and label batch
                next_batch = self.sess.run(next_element)
                batch_x = next_batch[0] # Image
                batch_y = next_batch[1] # Label
                
                n_per_batch = len(batch_x)
                
                feed = {self.x: batch_x, self.y: batch_y, self.keep_prob: 1 }
                cost, landmarks = self.sess.run([self.loss2, self.y_], feed_dict=feed)
                landmarks = np.reshape(landmarks, [-1, self.trunc_time_steps, 2, 168])

                predictions = np.append(predictions, landmarks, axis=0)
                total_loss += cost * n_per_batch
                total_data += n_per_batch
                
                print ("\rRead %d rows: [%-30s], batch loss %.3f | Elapsed: %.2f secs." % (total_data, '='*(i//5), cost, time.time()-start_loop), end='')
                i+=1 # this is just used for the progress bar

        except tf.errors.OutOfRangeError:
            # Without .repeat(), iterator is exhaustive. This is a common practice
            # If we want to use repeat, then we need to specify the number of batch, instead of using 'while' loop
            print ("\rRead %d rows: [%-30s], batch loss %.3f | Elapsed: %.2f secs." % (total_data, '='*30, cost, time.time()-start_loop), end='')
            pass

        # calculate the avg loss per epoch
        avg_cost = total_loss / total_data
            
        end_loop = time.time()
        msg = "\n{}\t- Total Loss: {:.3f}, time elapsed  : {:.2f} seconds\n".format("Test", avg_cost, end_loop-start_loop)
        log.info(msg)
        
        return predictions, avg_cost

    def run_epoch(self, epoch_idx, next_element, _current_state, is_training=False):
        """
            Run a single epoch (for validation or training).
            Running an epoch means looping through the whole batch iterator.
            The batch will be exhausted once this run is finished. 
            Note: iterator must be initialized first outside of this function.
        """
        start_loop = time.time()

        total_loss = 0
        total_rr_loss = 0
        total_cc_loss = 0
        total_data = 0

        try:
            i = 0
            while True:
                # Get input and label batch
                next_batch = self.sess.run(next_element)
                batch_x = next_batch[0] # Image
                batch_y = next_batch[1] # Label
                
                n_per_batch = len(batch_x)
                
                if is_training:
                    # Feed the network and optimize
                    feed = {self.x: batch_x, self.y: batch_y, self.init_state: _current_state, self.keep_prob: self.training_keep_prob }
                    _, merged_summ, cost, rr_err, cc_err, _current_state = self.sess.run([self.train_op, self.merged, self.loss2, self.rr_error, self.cc_error, self.states], feed_dict=feed)
                else:
                    # Feed it to the network
                    feed = {self.x: batch_x, self.y: batch_y, self.init_state: _current_state, self.keep_prob: 1 }
                    cost, rr_err, cc_err, _current_state = self.sess.run([self.loss2, self.rr_error, self.cc_error, self.states], feed_dict=feed)

                total_loss += cost * n_per_batch
                total_rr_loss += rr_err * n_per_batch
                total_cc_loss += cc_err * n_per_batch
                total_data += n_per_batch
                
                print ("\rRead %d rows: [%-30s], batch loss %.3f | Elapsed: %.2f secs." % (total_data, '='*(i//5), cost, time.time()-start_loop), end='')
                i+=1 # this is just used for the progress bar

        except tf.errors.OutOfRangeError:
            # Without .repeat(), iterator is exhaustive. This is a common practice
            # If we want to use repeat, then we need to specify the number of batch, instead of using 'while' loop
            print ("\rRead %d rows: [%-30s], batch loss %.3f | Elapsed: %.2f secs." % (total_data, '='*30, cost, time.time()-start_loop), end='')
            pass

        # calculate the avg loss per epoch
        avg_cost = total_loss / total_data   
        avg_rr_err  = total_rr_loss / total_data
        avg_cc_err  = total_cc_loss / total_data
            
        end_loop = time.time()

        # Log and summary
        summary = tf.Summary()
        summary.value.add(tag='{}/Loss'.format(self.unique_model_name), simple_value=avg_cost)
        summary.value.add(tag='{}/Weighted_RR_error'.format(self.unique_model_name), simple_value=avg_rr_err)
        summary.value.add(tag='{}/Weighted_CC_error'.format(self.unique_model_name), simple_value=avg_cc_err)
        
        if is_training:
            epoch_name = "Training"
            self.train_writer.add_summary(summary, epoch_idx)
            self.train_writer.add_summary(merged_summ, epoch_idx) # standard merged summ
        else:
            epoch_name = "Validation"
            self.val_writer.add_summary(summary, epoch_idx)

        msg = "\n{} {}\t- Total Loss: {:.3f}, RR_loss: {:.3f}, CC_loss: {:.3f}, time elapsed  : {:.2f} seconds\n".format(epoch_idx+1, epoch_name, avg_cost, avg_rr_err, avg_cc_err, end_loop-start_loop)
        log.info(msg)
        
        return avg_cost, _current_state