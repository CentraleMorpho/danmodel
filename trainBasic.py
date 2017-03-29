import vgg as model
import tensorflow as tf
import numpy as np
import time
import pickle
import cv2
	
	
def train(lr=0.0001,
          nb_iterations=1000000,
          batch_size=64):


    with tf.Graph().as_default():
		

		images, labels = model.labeled_inputs('training', batch_size)
		imagesVal, labelsVal = model.labeled_inputs('test', batch_size)

		logits = model.inference_vgg(images, None, training=True)
		
		objectiveGT = model.loss_op(logits, labels, batch_size)
		
		accuracy = model.evaluate(logits, labels, 10.)
		
		logitsVal = model.inference_vgg(imagesVal, True, training=False)
		objectiveGTVal = model.loss_op(logitsVal, labelsVal, batch_size)
		accuracyVal = model.evaluate(logitsVal, labelsVal, 10.)
		
		optimizer = tf.train.AdamOptimizer(lr)
		global_step = tf.Variable(0, name="global_step", trainable=False)
		train_step_GT = optimizer.minimize(objectiveGT, global_step=global_step)
		
		summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
		summaries.append(tf.scalar_summary('Loss Training GT', objectiveGT))
		summaries.append(tf.scalar_summary('Loss Validation GT', objectiveGTVal))
		summary_op = tf.merge_summary(summaries)
		


		# Start running operations on the Graph.
		with tf.Session() as sess:
		
			train_writer = tf.train.SummaryWriter('../summaries' + '/B73Basic', sess.graph)
			
			saver = tf.train.Saver()
			sess.run(tf.initialize_all_variables())
			sess.run(tf.initialize_local_variables())

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)

			model.init()


			for iteration in range(nb_iterations):
				
				
				
				if iteration%100==0:
					result = sess.run([logits, objectiveGT, train_step_GT,accuracy,objectiveGTVal,accuracyVal,summary_op])		
					trn_lossGT = result[1]
					acc=result[3]
					val_lossGT=result[4]
					accval = result[5]

					print("iter:%5d, trn_lossGT: %s, acc : %s, val_lossGT : %s, val_acc : %s"  % (iteration, trn_lossGT, acc, val_lossGT, accval))
			
					#Save to summaries
					summary = tf.Summary()
					summary.ParseFromString(result[6])
					train_writer.add_summary(summary, iteration)
				else:
					result = sess.run([logits, objectiveGT, train_step_GT])	
				
					
				# Save the model checkpoint periodically.
				if iteration%2000==0:
					print("Saving model...")
					saver.save(sess, "models/B73Basic.cpkt", global_step=iteration)
					print("Model saved")
					
				
                    


if __name__ == '__main__':
    batch_size = 64
    train(batch_size = batch_size)
