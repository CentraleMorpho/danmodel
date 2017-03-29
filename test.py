import danmodel as model
import tensorflow as tf
import numpy as np
import time
import pickle


def test(nb_iterations=46,
          batch_size=64):


    with tf.Graph().as_default():

		imagesVal, labelsVal = model.labeled_inputs('test', batch_size)
		featuresVal = model.feature_extractor(imagesVal, None)
		logitsVal = model.label_predictor(featuresVal,None)

		accuracyVal = model.evaluate(logitsVal, labelsVal, 5.)
		accuracy10Val = model.evaluate(logitsVal, labelsVal, 10.)
		
		accuracys5 = np.zeros([1,3], dtype=float) # Calculate mean of accuracys
		accuracys10 = np.zeros([1,3], dtype=float) 
		mae = np.zeros([1,3], dtype=float)
		
		# Start running operations on the Graph.
		with tf.Session() as sess:
			saver = tf.train.Saver()
			saver.restore(sess,"models/B73Basic.cpkt-328000")
			print("Model restored")

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)

			model.init()


			for iteration in range(nb_iterations):
				result = sess.run(
					[imagesVal, labelsVal, logitsVal, accuracyVal, accuracy10Val], 
					) 

				trn_acc = result[3]
				trn_acc10 = result[4]
				#print("iter:%5d, VALIDATION BATCH, precisions YPR : %s, %s, %s" % (iteration, trn_acc[0], trn_acc[1], trn_acc[2]))
				accuracys5[0,0]+=trn_acc[0]
				accuracys5[0,1]+=trn_acc[1]
				accuracys5[0,2]+=trn_acc[2]
				
				accuracys10[0,0]+=trn_acc10[0]
				accuracys10[0,1]+=trn_acc10[1]
				accuracys10[0,2]+=trn_acc10[2]
		
				
				labels = result[1]
				logits=result[2]
				mae+=np.mean(abs(logits-labels),axis=0)
        
		accuracys5/=nb_iterations
		accuracys10/=nb_iterations
		mae/=nb_iterations
		mae/=64
		mae*=180
		print("accuracys 5 degrees validation set : %s, %s, %s" %(accuracys5[0,0],accuracys5[0,1],accuracys5[0,2]))
		print("accuracys 10 degrees validation set : %s, %s, %s" %(accuracys10[0,0],accuracys10[0,1],accuracys10[0,2]))
		print("MAE %s" %(mae))

if __name__ == '__main__':
    test()
