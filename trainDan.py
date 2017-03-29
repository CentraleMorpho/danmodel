import danmodel as model
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

		features_labeled = model.feature_extractor(images, None)
		d_logits_labeled = model.domain_predictor(features_labeled,None)
		domain_labeled_all1 = tf.ones(tf.shape(d_logits_labeled))
		domain_labeled = tf.mul(domain_labeled_all1, tf.constant([1.0,0.0]))
		
		logits = model.label_predictor(features_labeled,None, training=True)
		objectiveGT = model.loss_op(logits, labels, batch_size)
		accuracy = model.evaluate(logits, labels, 10.)
		
		features_val = model.feature_extractor(imagesVal, True)
		logitsVal = model.label_predictor(features_val, True, training=False)
		objectiveGTVal = model.loss_op(logitsVal, labelsVal, batch_size)
		accuracyVal = model.evaluate(logitsVal, labelsVal, 10.)

		#OTHER DOMAIN :

		imagesFlop_nonoise = model.flop_inputs(batch_size)
		imagesFlop = model.noise_batch(imagesFlop_nonoise)

		features_unlabeled = model.feature_extractor(imagesFlop, True)
		d_logits_unlabeled = model.domain_predictor(features_unlabeled, True)
		domain_unlabeled_all1 = tf.ones(tf.shape(d_logits_unlabeled))
		domain_unlabeled = tf.mul(domain_unlabeled_all1, tf.constant([0.0,1.0]))
		
		#Loss domain
		domains = tf.concat(0,[domain_labeled,domain_unlabeled])
		d_logits = tf.concat(0,[d_logits_labeled,d_logits_unlabeled])
		objectiveDomain = model.loss_domain(d_logits,domains)
		
		#FLOP
		#imagesFlop2_nonoise = tf.reverse(imagesFlop_nonoise, [False,False,True,False])
		#imagesFlop2 = model.noise_batch(imagesFlop2_nonoise)
		
		#features_unlabeled2 = model.feature_extractor(imagesFlop2, True)
		
		logits_unlabeled = model.label_predictor(features_unlabeled, True, training=False)
		#logits_unlabeled2 = model.label_predictor(features_unlabeled2, True, training=False)
		
		#objectiveFlop = model.loss_flop(logits_unlabeled, logits_unlabeled2,batch_size) 
		
		
		
		#Total loss
		#objectiveTot = tf.add(10*objectiveGT,objectiveFlop)
		objectiveTot = 10*objectiveGT
		
		#Loss CNN random classifier
		objectiveCNN = tf.abs(objectiveDomain-0.693147)
		scope_name_list = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2"]
		var_name_cnn_to_update = [] # contains the list of variables we want to update using the objectiveCNN loss
		for scope in scope_name_list:
			var_name_cnn_to_update = var_name_cnn_to_update + tf.get_collection(tf.GraphKeys.VARIABLES, scope)
		print(var_name_cnn_to_update)
		
		#Variables of the classifier
		scope_name_list = ["d_fc1","d_fc2","d_fc3"]
		var_name_classifier_to_update = [] # contains the list of variables we want to update using the objectiveCNN loss
		for scope in scope_name_list:
			var_name_classifier_to_update = var_name_classifier_to_update + tf.get_collection(tf.GraphKeys.VARIABLES, scope)
		print(var_name_classifier_to_update)
		###########################
		
		optimizer = tf.train.AdamOptimizer(lr)
		global_step = tf.Variable(0, name="global_step", trainable=False)
		train_step = optimizer.minimize(objectiveTot, global_step=global_step)
		train_stepCNN = optimizer.minimize(2*objectiveCNN, global_step=global_step, var_list = var_name_cnn_to_update)
		train_stepDomain = optimizer.minimize(objectiveDomain, global_step=global_step, var_list = var_name_classifier_to_update)
		
		summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
		summaries.append(tf.scalar_summary('Loss Training GT', objectiveGT))
		summaries.append(tf.scalar_summary('Loss Training Domain', objectiveDomain))
		#summaries.append(tf.scalar_summary('Loss Training Flop', objectiveFlop))
		summaries.append(tf.scalar_summary('Loss Validation GT', objectiveGTVal))
		summary_op = tf.merge_summary(summaries)
		

		with tf.Session() as sess:
		
			train_writer = tf.train.SummaryWriter('../summaries' + '/DANibug_B73_NoFlop_lossGT10_lossCNN2_noDropOut', sess.graph)
			
			sess.run(tf.initialize_all_variables())
			sess.run(tf.initialize_local_variables())
			
			#saver = tf.train.Saver(tf.all_variables())
			
			#RESTORING SUBSET OF MODEL
			scope_name_list = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1", "conv3_2","fc6","fc8"]
			basic_model_variables = [] 
			for scope in scope_name_list:
				basic_model_variables = basic_model_variables + tf.get_collection(tf.GraphKeys.VARIABLES, scope)
			saver = tf.train.Saver(var_list = basic_model_variables)
			saver.restore(sess,"models/B73Basic.cpkt-444000")
			#saver = tf.train.Saver()
			#sess.run(tf.initialize_all_variables())
			#sess.run(tf.initialize_local_variables())

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)

			model.init()


			for iteration in range(nb_iterations):
				
				
				
				if iteration%100==0:
					result = sess.run([objectiveGT, objectiveDomain, train_step,objectiveGTVal,summary_op, objectiveCNN, train_stepCNN, train_stepDomain])		
					trn_lossGT = result[0]
					#trn_lossFlop=result[1]
					trn_lossCNN=result[5]
					trn_lossDomain=result[1]
					val_lossGT = result[3]
					

					#print("iter:%5d, trn_lossGT: %s, trn_lossFlop : %s, trn_lossDomain : %s, trn_lossCNN : %s, val_lossGT : %s" % (iteration, trn_lossGT, trn_lossFlop, trn_lossDomain, trn_lossCNN, val_lossGT))
					print("iter:%5d, trn_lossGT: %s, trn_lossDomain : %s, val_lossGT : %s" % (iteration, trn_lossGT, trn_lossDomain, val_lossGT))
			
					#Save to summaries
					summary = tf.Summary()
					summary.ParseFromString(result[4])
					train_writer.add_summary(summary, iteration)
				else:
					result = sess.run([train_step, train_stepCNN, train_stepDomain])	
				
				# Save the model checkpoint periodically.
				if iteration%2000==0:
					print("Saving model...")
					saver.save(sess, "models/DANibug_B73_NoFlop_lossGT10_lossCNN2_noDropOut.cpkt", global_step=iteration)
					print("Model saved")
					
				
                    


if __name__ == '__main__':
    batch_size = 64
    train(batch_size = batch_size)
