import tensorflow as tf
import numpy as np
import h5py
import os

class NetworkModelCNN:
    output_size = 4
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
        
        graph = tf.get_default_graph()
        self.placeholder_x = graph.get_tensor_by_name("x1:0")
        self.prediction = graph.get_tensor_by_name("y_:0")
        
        self.is_training = graph.get_tensor_by_name("is_training:0")
        self.keep_prob = graph.get_tensor_by_name("keep_prob:0")


    def predict_corners(self, ed_frame):
        feed_dict = { self.placeholder_x: ed_frame, self.keep_prob: 1, self.is_training: False }
        corners = self.session.run(self.prediction, feed_dict)
        
        return corners    