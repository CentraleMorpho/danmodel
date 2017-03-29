import danmodel as model
import tensorflow as tf
import numpy as np
import time
import pickle


def createDBLabeled(nb_iterations=100,
          batch_size=64):

	model.init()
	dbLabeled = np.zeros((64*nb_iterations,1600))
	with tf.Graph().as_default():

		images, labels = model.labeled_inputs('training', batch_size)
		features_labeled = model.feature_extractor(images, None)
		
		# Start running operations on the Graph.
		with tf.Session() as sess:
			saver = tf.train.Saver()
			saver.restore(sess,"models/DAN_Flop_pretrained_lossDan1_objectiveCNN1_manyiter.cpkt-998000")
			print("Model restored")

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)

			model.init()


			for iteration in range(nb_iterations):
				result = sess.run(
					[features_labeled], 
					) 
				dbLabeled[64*iteration:64*(iteration+1),0:1600]=result[0]
		
	np.savetxt("dbLabeled_DAN_Flop_pretrained_lossDan1_objectiveCNN1_manyiter.csv",dbLabeled,delimiter=",")

if __name__ == '__main__':
    createDBLabeled()
