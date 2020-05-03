import os
import time

import h5py
import numpy as np
import tensorflow as tf

class NetworkModelRNNCNN:
    time_steps = 20
    output_size = 2 * 168
    session = None
    
    def __del__(self):
        if self.session is not None:
            print('Closing tf.session...')
            self.session.close()

    def __init__(self, model_dir, model_name):
        self.network_name = model_name

        # ----- tensorflow stuff -------------
        tf.reset_default_graph() 

        if self.session is None:
            print('\nCreating new tf.session')
            self.session = tf.Session()

        print('Restoring model {}'.format(model_name))
        new_saver = tf.train.import_meta_graph('{}/{}.meta'.format(model_dir, model_name))
        new_saver.restore(self.session, tf.train.latest_checkpoint(model_dir))
        
        # restore the tensors
        graph = tf.get_default_graph()
        self.placeholder_x = graph.get_tensor_by_name("x:0")
        self.prediction = graph.get_tensor_by_name("y_:0")
        self.keep_prob = graph.get_tensor_by_name("keep_prob:0")
        

    def predict_landmark_sequences(self, image_sequences):
        # make sure the data are in correct shape
        X_data = image_sequences[:,0:self.time_steps]
        
        feed_dict = { self.placeholder_x: X_data, self.keep_prob: 1}
        res = self.session.run(self.prediction, feed_dict)
        points = np.reshape(res, [-1, self.time_steps, 2,168])
        
        return points