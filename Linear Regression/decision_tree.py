import numpy as np
import matplotlib.pyplot as plt
import argparse

clean_dataset = np.loadtxt("wifi_db/clean_dataset.txt")
noisy_dataset = np.loadtxt("wifi_db/noisy_dataset.txt")

def calc_information_gain(dataset, l_dataset, r_dataset):
	"""
	Computes the information gain when splitting a dataset into
	two. The formula used is:

	Gain(S_all, S_left, S_right) = H(S_all) - Remainder(S_left, S_right)	
	"""
	return calc_entropy(dataset) - calc_remainder(l_dataset, r_dataset)

def calc_entropy(dataset):
	"""
	Calculates the entropy of a dataset. The formula used is:

	H(dataset) = - sum{1..K}(p_k * log2(p_k))
	"""
	labels = np.unique(dataset)
	entropy = 0
	for label in labels:
		p_k = len(dataset[dataset == label]) / len(dataset)
		entropy += -p_k * np.log2(p_k)
	return entropy

def calc_remainder(l_dataset, r_dataset):
	"""
	Calculates the remainder for two datasets. The formula used is:

	Remainder(S_left, S_right) = (|S_left| / (|S_left| + |S_right|)) * H(S_left) + (|S_right| / (|S_left| + |S_right|))
	"""
	weight_l = len(l_dataset) / (len(l_dataset) + len(r_dataset))
	weight_r = len(r_dataset) / (len(l_dataset) + len(r_dataset))
	return weight_l * calc_entropy(l_dataset) + weight_r * calc_entropy(r_dataset)

def create_node(attribute=None, value=None, left=None, right=None, leaf=False):
	"""
	Creates a node using a dictionary with the following key-value pairs:
	Attribute, Value, Left Dataset, Right Dataset, Leaf Boolean Flag
	"""
	return { "attribute" : attribute, "value" : value, "left" : left, "right" : right, "leaf": leaf }

def get_unique_labels(labels):
	"""
	Gets all the unique labels
	"""
	return np.unique(labels)

def split_dataset(dataset, attribute, value):
	"""
	Splits a dataset set into two using an attribute and compares the value.
	"""
	l_dataset = np.array([r for r in dataset if r[attribute] < value])
	r_dataset = np.array([r for r in dataset if r[attribute] >= value])
	return l_dataset, r_dataset

def find_split(dataset, total_samples, num_of_attributes):
	"""
	Finds the optimal splitting point and returns a dictionary containing the
	following: attribute, value, left dataset, right dataset.
	"""
	split = {}
	highest_info_gain = -float("inf")

	for attribute in range(num_of_attributes):
		sorted_attributes = np.sort(dataset[:, attribute])
		for x1, x2 in zip(sorted_attributes, sorted_attributes[1:]):
			if x1 == x2:
				continue
			l_dataset, r_dataset = split_dataset(dataset, attribute, x2)
			if len(l_dataset) <= 0 or len(r_dataset) <= 0:
				continue
			parent = dataset[:, -1]
			left = l_dataset[:, -1]
			right = r_dataset[:, -1]
			info_gain = calc_information_gain(parent, left, right)
			if info_gain > highest_info_gain:
				mid_point = (x1 + x2) / 2.0
				split = { "attribute" : attribute,
						"value" : mid_point,
						"l_dataset" : l_dataset, 
						"r_dataset" : r_dataset }
				highest_info_gain = info_gain

	return split

def decision_tree_learning(training_dataset, depth):
	""" 
	Builds a decision tree given a dataset. Returns a
	node(leaf or decision) along with the current depth of the tree.
	"""
	x, labels = training_dataset[:, :-1], training_dataset[:, -1]
	total_samples, num_of_attributes = np.shape(x)
	unique_labels = get_unique_labels(labels)
	if len(unique_labels) == 1:
		return (create_node(value=unique_labels[0], leaf=True), depth)
	else:
		split = find_split(training_dataset, total_samples, num_of_attributes)
		l_branch, l_depth = decision_tree_learning(split["l_dataset"], depth + 1)
		r_branch, r_depth = decision_tree_learning(split["r_dataset"], depth + 1)
		node = create_node(split["attribute"], split["value"], l_branch, r_branch)
		return (node, max(l_depth, r_depth))

def build_figure(node, current_depth, depth, width, x, y, graph):
	"""
	Traverses tree and plots tree onto graph.
	"""
	if node["leaf"]:
		text = "leaf {value}".format(value=node["value"])
		graph.text(x, y, text, color="blue", bbox=dict(facecolor="white", edgecolor="blue"))
	else:
		text = "X{i} < {value}".format(i=node["attribute"], value=node["value"])
		graph.text(x, y, text, color="blue", bbox=dict(facecolor="white", edgecolor="blue"))
		dist = width / (current_depth + 1)
		if node["left"]:
			build_figure(node["left"], current_depth + 1, depth, width, x - dist, y - 10, graph)
			graph.plot([x, x - dist], [y, y - 10])
		if node["right"]:
			build_figure(node["right"], current_depth + 1, depth, width, x + dist, y - 10, graph)
			graph.plot([x, x + dist], [y, y - 10])

def visualise_tree(root, depth):
	"""
	Plots tree onto a graph and saves the graph to a file.
	"""
	width = 500
	fig, subplot = plt.subplots(figsize=(150, 30))
	subplot.set_title("Decision Tree")
	build_figure(root, 0, depth, width, 0, 0, subplot)
	plt.ylim([3, -depth * 10])
	plt.gca().invert_yaxis()
	subplot.axis("off")
	plt.savefig("images/decision_tree.png", dpi=80)

def get_prediction(x, node):
	"""
	Traverses decision tree and finds the predicted value.
	"""
	if node["leaf"]:
		return node["value"]
	else:
		attribute_value = x[node["attribute"]]
		if attribute_value < node["value"]:
			return get_prediction(x, node["left"])
		else:
			return get_prediction(x, node["right"])

def create_confusion_matrix(labels, predictions):
	"""
	Creates a confusion matrix using the actual labels and predicted labels.
	"""
	classes = get_unique_labels(labels)
	num_of_classes = len(classes)
	matrix = np.zeros((num_of_classes, num_of_classes))
	for i in range(num_of_classes):
		for j in range(num_of_classes):
			actual_match = labels == classes[i]
			predicted_match = predictions == classes[j]
			matrix[i, j] = np.sum(actual_match & predicted_match)
	return matrix

def calc_accuracy(confusion_matrix):
	"""
	Computes the accuracy of a model using the confusion matrix.
	"""
	total = np.sum(confusion_matrix)
	diagonal_sum = np.trace(confusion_matrix)
	return diagonal_sum / total

def calc_recall(confusion_matrix):
	"""
	Computes the recall of a model using the confusion matrix.
	"""
	diagonal = np.diag(confusion_matrix)
	row_sum = np.sum(confusion_matrix, axis=1)
	return np.divide(diagonal, row_sum)

def calc_precision(confusion_matrix):
	"""
	Computes the precision of a model using the confusion matrix.
	"""
	diagonal = np.diag(confusion_matrix)
	column_sum = np.sum(confusion_matrix, axis=0)
	return np.divide(diagonal, column_sum)

def calc_f1(recalls, precisions):
	"""
	Computes the f1 of a model using the recall and precision.
	"""
	return (2 * precisions * recalls) / (precisions + recalls)

def split_train_test(dataset, start, end, N):
	"""
	Splits the dataset into test and training+validation.
	"""
	np.random.shuffle(dataset)
	test_split = dataset[start : end]
	training_data = np.concatenate((dataset[0 : start], dataset[end:N]), axis=0)
	x_test = test_split[:, :-1]
	y_test = test_split[:, -1]
	return (training_data, x_test, y_test)

def generate_evaluation_info(confusion_matrix, depths_before_pruning=None, depths_after_pruning=None):
	"""
	Computes evaluation metrics using confusion matrix.
	"""
	evaluation_info = {}
	evaluation_info["accuracy"] = calc_accuracy(confusion_matrix)
	evaluation_info["confusion_matrix"] = confusion_matrix
	evaluation_info["recall"] = calc_recall(confusion_matrix)
	evaluation_info["precision"] = calc_precision(confusion_matrix)
	evaluation_info["f1"] = calc_f1(evaluation_info["recall"], evaluation_info["precision"])
	if depths_before_pruning:
		evaluation_info["avg_depth_before_pruning"] = np.mean(depths_before_pruning)
	if depths_after_pruning:
		evaluation_info["avg_depth_after_pruning"] = np.mean(depths_after_pruning)
	return evaluation_info

def evaluate_cross_validation(dataset, folds=10):
	"""
	Evaluates a dataset using cross-validation.
	"""
	N = len(dataset)
	fold_indices = np.linspace(0, N + 1, folds + 1, dtype=int)
	confusion_matrices = []

	for i in range(len(fold_indices) - 1):
		# Split Into Test + Training Data
		training_data, x_test, y_test = split_train_test(dataset, fold_indices[i], fold_indices[i + 1], N)
		# Train Model Using Training Data
		trained_tree, depth = decision_tree_learning(training_data, 0)
		# Predict Labels Using Test Data
		predictions = [get_prediction(x, trained_tree) for x in x_test]
		# Create Confusion Matrix Using Actual & Predicted
		confusion_matrix = create_confusion_matrix(y_test, predictions)
		confusion_matrices.append(confusion_matrix)

	avg_confusion_matrix = np.mean(np.array(confusion_matrices), axis=0)

	return generate_evaluation_info(avg_confusion_matrix)
	
def evaluate_nested_cross_validation(dataset, folds=10, prune=False):
	"""
	Evaluates a dataset using nested cross-validation.
	"""
	N = len(dataset)
	fold_indices = np.linspace(0, N + 1, folds + 1, dtype=int)
	confusion_matrices = []
	depths_before_pruning = []
	depths_after_pruning = []
	for i in range(len(fold_indices) - 1):
		# Split Into Test + Training/Validation Data
		training_data, x_test, y_test = split_train_test(dataset, fold_indices[i], fold_indices[i + 1], N)
		M = len(training_data)
		nested_fold_indices = np.linspace(0, M + 1, (folds - 1) + 1, dtype=int)
		nested_depths_before_pruning = []
		nested_depths_after_pruning = []
		nested_confusion_matrices = []
		for j in range(len(nested_fold_indices) - 1):
			# Split Into Validation + Training Data
			training_data, x_validation, y_validation = split_train_test(dataset, nested_fold_indices[j], nested_fold_indices[j + 1], M)
			# Train Model Using Training Data
			trained_tree, depth = decision_tree_learning(training_data, 0)
			nested_depths_before_pruning.append(depth)
			# Prune Decision Tree
			if prune:
				pruned_tree, pruned_depth = prune_tree(training_data, x_validation, y_validation, trained_tree, 0)
				prev_validation_error = compute_validation_error(x_validation, y_validation, pruned_tree)
				while True:
					pruned_tree, pruned_depth = prune_tree(training_data, x_validation, y_validation, pruned_tree, 0)
					current_validation_error = compute_validation_error(x_validation, y_validation, pruned_tree)
					if prev_validation_error == current_validation_error:
						break
				nested_depths_after_pruning.append(pruned_depth)
				trained_tree = pruned_tree
			# Predict Labels Using Test Data
			predictions = [get_prediction(x, trained_tree) for x in x_test]
			# Create Confusion Matrix Using Actual & Predicted
			confusion_matrix = create_confusion_matrix(y_test, predictions)
			nested_confusion_matrices.append(confusion_matrix)
		
		confusion_matrices.append(np.mean(np.array(nested_confusion_matrices), axis=0))
		depths_before_pruning.append(np.mean(np.array(nested_depths_before_pruning)))
		if prune:
			depths_after_pruning.append(np.mean(np.array(nested_depths_after_pruning)))

	avg_confusion_matrix = np.mean(np.array(confusion_matrices), axis=0)

	return generate_evaluation_info(avg_confusion_matrix, depths_before_pruning, depths_after_pruning if prune else None)

def print_evaluation_info(evaluation_info):
	"""
	Prints out the evaluation metrics.
	"""
	accuracy = evaluation_info["accuracy"]
	confusion_matrix = evaluation_info["confusion_matrix"]
	recall = evaluation_info["recall"]
	precision = evaluation_info["precision"]
	f1 = evaluation_info["f1"]

	def print_dashed_line():
		print("------------------------------------")

	print_dashed_line()
	print("Accuracy: {accuracy}".format(accuracy=evaluation_info["accuracy"]))
	print_dashed_line()
	print("Recalls: {recall}".format(recall=evaluation_info["recall"]))
	print_dashed_line()
	print("Precision: {precision}".format(precision=evaluation_info["precision"]))
	print_dashed_line()
	print("F1: {f1}".format(f1=evaluation_info["f1"]))
	print_dashed_line()
	print("Confusion Matrix:\n{confusion_matrix}".format(confusion_matrix=evaluation_info["confusion_matrix"]))
	print_dashed_line()
	if "avg_depth_before_pruning" in evaluation_info:
		print("Average Depth Before Pruning: {depth}".format(depth=evaluation_info["avg_depth_before_pruning"]))
		print_dashed_line()
	if "avg_depth_after_pruning" in evaluation_info:
		print("Average Depth After Pruning: {depth}".format(depth=evaluation_info["avg_depth_after_pruning"]))
		print_dashed_line()

def is_internal_node(node):
	"""
	Checks if a node is an internal node.
	"""
	return node["right"]["leaf"] and node["left"]["leaf"]

def compute_validation_error(x_validation, y_validation, node):
	"""
	Computes the validation error using the validation dataset.
	"""
	predictions = [get_prediction(x, node) for x in x_validation]
	validation_error = np.sum(predictions != y_validation)
	return validation_error

def get_majority_class_label(dataset):
	"""
	Gets the majority class label
	"""
	labels = dataset[:, -1]
	unique, counts = np.unique(labels, return_counts=True)
	return unique[np.argmax(counts)]

def prune_tree(training_data, x_validation, y_validation, node, depth):
	"""
	1. Go through each internal node that are connected only to leaf nodes
	2. Turn each into a leaf node (with majority class label)
	3. Evaluate pruned tree on validation set. Prune if accuracy higher than unpruned
	4. Repeat untill all such nodes have been tested
	"""
	if node["leaf"]:
		return (node, depth)
	else:
		l_dataset, r_dataset = split_dataset(training_data, node["attribute"], node["value"])

		node["left"], l_depth = prune_tree(l_dataset, x_validation, y_validation, node["left"], depth + 1)
		node["right"], r_depth = prune_tree(r_dataset, x_validation, y_validation, node["right"], depth + 1)

		if is_internal_node(node):
			x1 = compute_validation_error(x_validation, y_validation, node)
			majority_class_label = get_majority_class_label(training_data)
			x2 = np.sum(y_validation != majority_class_label)
			if x1 > x2:
				leaf_node = create_node(value=majority_class_label, leaf=True)
				return (leaf_node, max(l_depth, r_depth) - 1)

		return (node, max(l_depth, r_depth))
		
def main():
	parser = argparse.ArgumentParser("decision_tree")
	parser.add_argument("-c", "--clean-dataset", action='store_true', help="Uses clean dataset (default)")
	parser.add_argument("-n", "--noisy-dataset", action='store_true', help="Uses noisy dataset")
	parser.add_argument("-v", "--visualise", action='store_true', help="Visualises decision tree and stores to file")
	parser.add_argument("-cv", "--evaluate-cross-validation", action='store_true', help="Evaluates decision tree using cross-validation (without pruning)")
	parser.add_argument("-ncv", "--evaluate-nested-cross-validation", action='store_true', help="Evaluates decision tree using nested cross-validation")
	parser.add_argument("-p", "--prune", action='store_true', help="Enables pruning")

	args = parser.parse_args()

	dataset = clean_dataset if not args.noisy_dataset else noisy_dataset
	
	if args.visualise:
		tree, depth = decision_tree_learning(dataset, 0)
		visualise_tree(tree, depth)
	elif args.evaluate_cross_validation:
		evaluation_info = evaluate_cross_validation(dataset)
		print_evaluation_info(evaluation_info)
	elif args.evaluate_nested_cross_validation:
		evaluation_info = evaluate_nested_cross_validation(dataset, prune=args.prune)
		print_evaluation_info(evaluation_info)

if __name__ == "__main__":
	main()