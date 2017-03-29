import os
import random
import math
import tensorflow as tf
from pylab import *
import numpy as np
import danmodel as model
import cv2

IMAGE_SIZE=39

def writePredsOnImage(images,preds):
	imagesWritten = np.zeros((images.shape[0],512,512,1))
	for j in range(0,images.shape[0]):
				image = cv2.resize(images[j,:],(512,512), interpolation = cv2.INTER_CUBIC)
				txt = ' '.join(['%.1f'%(f*180) for f in preds[j,:]])
				cv2.putText(image,txt,(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
				#txt = ' '.join(['%.1f'%(f*180) for f in gt[j,:]])
				#cv2.putText(image,txt,(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
				imagesWritten[j,:,:,0]=image
	imagesWritten = np.ndarray.astype(imagesWritten, float32)
	return imagesWritten


if __name__=='__main__':
	batch_size=10
	with tf.Graph().as_default():
		
		imagesFlop_nonoise = model.flop_inputs(batch_size)
		#images, labels = model.labeled_inputs('training', batch_size)
		
		imagesFlop2_nonoise = tf.reverse(imagesFlop_nonoise, [False,False,True,False])
		imagesFlop = model.noise_batch(imagesFlop_nonoise)
		imagesFlop2 = model.noise_batch(imagesFlop2_nonoise)
		

		features_unlabeled = model.feature_extractor(imagesFlop, None)
		features_unlabeled2 = model.feature_extractor(imagesFlop2, True)
		#features = model.feature_extractor(images, None)
		
		logitsFlop = model.label_predictor(features_unlabeled, None, training=False)
		logitsFlop2 = model.label_predictor(features_unlabeled2, True, training=False)
		#logits = model.label_predictor(features, None, training=False)
		
		imagesWritten = tf.py_func(writePredsOnImage,[imagesFlop,logitsFlop],[tf.float32])
		imagesWritten = tf.convert_to_tensor(imagesWritten, dtype = tf.float32)
		imagesWritten = tf.reshape(imagesWritten,[batch_size,512,512,1])
		
		summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
		summary = tf.image_summary('images',imagesWritten, max_images=40)
		summaries.append(summary)
		#summary = tf.image_summary('images_nonoise',imagesFlop_nonoise, max_images=40)
		#summaries.append(summary)
		
		#summary = tf.image_summary('imagesLabeled',images, max_images=40)
		#summaries.append(summary)
		
		summary_op = tf.merge_summary(summaries)
		
		init_op = tf.initialize_all_variables()
		with tf.Session() as sess:
			saver = tf.train.Saver()
			saver.restore(sess,"models/DAN_B73_Flop_pretrained_lossGT10_lossCNN3.cpkt-54000")
			print("Model restored")
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			
			model.init()
		  
			writer = tf.train.SummaryWriter('../summaries' + '/testestimation_B106_B73_GT10_CNN3', sess.graph)

			for i in range(1): 
				result=sess.run([imagesFlop,summary_op,logitsFlop])
				writer.add_summary(result[1])

			writer.close()

			coord.request_stop()
			coord.join(threads)

	
