import pandas
from string_builder import StringBuilder
from perceptron import Perceptron
from standard_perceptron import StandardPerceptron
from decay_perceptron import DecayPerceptron
from margin_perceptron import MarginPerceptron
from average_perceptron import AveragePerceptron
# from matplotlib import pyplot as plt

def rem_zero(df): df.iloc[:][0][df.iloc[:][0] == 0] = -1

def cross_validate(perceptron, eta, mew, sb, epochs=10):
	folds = { i:pandas.read_csv("hw2_data/Data/CVSplits/training%s.csv" % i, header=None) for i in range(0, 5, 1) }
	for i in folds.keys(): rem_zero(folds[i])

	for fold in folds.keys():

		train_accs = []
		test_accs = []

		indices = [i for i in range(0, 5, 1) if i != fold]
		train_set_fold = pandas.concat([folds[i] for i in indices])
		test_set_fold = folds[i]

		perceptron.perceptron(train_set_fold, epochs, eta, mew)

		train_accs.append(perceptron.accuracy(train_set_fold))
		test_accs.append(perceptron.accuracy(test_set_fold))

	sb.append_line("5-Fold CV Train Set Accuracy after %s Epochs (eta=%f, mew=%f): %f" % 
		(epochs, eta, mew, sum(train_accs) / len(train_accs)))
	sb.append_line("5-Fold CV Test Set Accuracy after %s Epochs (eta=%f, mew=%f): %f" % 
		(epochs, eta, mew, sum(test_accs) / len(test_accs)), True)

	return sum(test_accs) / len(test_accs)

def exec_experiments(perceptron, etas, train_df, test_df, dev_df, sb, mews=[0]): 
	sb.append_line("Experiments running for %s" % type(perceptron).__name__, True)

	best_eta = best_mew = best_acc = 0
	for eta in etas:
		for mew in mews:
			if (valid_acc := cross_validate(perceptron, eta, mew, sb)) > best_acc:
				best_acc = valid_acc
				best_eta = eta
				best_mew = mew

	sb.append_line("Best Hyperparameter Combination: eta=%f, mew=%f" % (best_eta, best_mew))
	sb.append_line("Best Hyperparameter Combination CV Test Set Accuracy: %f" % best_acc, True)

	per_epoch = perceptron.perceptron(train_df, 20, best_eta, best_mew)

	per_epoch_accs = []
	for i in range(len(per_epoch)):
		per_epoch_accs.append((acc := per_epoch[i].accuracy(dev_df)))
		sb.append_line("Accuracy on Dev Set after %s Epochs: %f" % (i + 1, acc))
	sb.append_sep()

	sb.append_line("Number of Weight Vector Updates made on Train Set: %s" % perceptron.updates, True)

	best_epoch = best_acc = 0
	for i in range(len(per_epoch_accs)):
		if (acc := per_epoch_accs[i]) > best_acc:
			best_acc = acc
			best_epoch = i

	sb.append_line("Best Epoch on Dev Data: %s" % 
		(best_epoch + 1))
	sb.append_line("Best Epoch Accuracy on Dev Data: %f" % 
		(per_epoch_accs[best_epoch]), True)

	sb.append_line("Best Perceptron(epoch=%s, eta=%f, mew=%s) Accuracy on Test Data: %f" % 
		(best_epoch + 1, best_eta, best_mew, per_epoch[best_epoch].accuracy(test_df)), True)

	# plt.plot(range(1, 21, 1), per_epoch_accs)
	# plt.xlabel("Epochs")
	# plt.ylabel("Dev Set Acc.")
	# plt.title("%s Per Epoch Acc.\neta=%f, mew=%f" % (type(perceptron).__name__, best_eta, best_mew))
	# plt.show()

	with open("%s.txt" % type(perceptron).__name__, "w") as f: sb.dump_output(f)

def main():	
	sb = StringBuilder()
	train_df = pandas.read_csv("hw2_data/Data/training_data.csv", header=None)
	test_df = pandas.read_csv("hw2_data/Data/testing_data.csv", header=None)
	dev_df = pandas.read_csv("hw2_data/Data/developmental_data.csv", header=None)
	rem_zero(train_df)
	rem_zero(test_df)
	rem_zero(dev_df)

	mode = train_df.iloc[:][0].mode()[0]
	mode_acc_test = sum([1 if res == mode else 0 for res in test_df.iloc[:][0]]) / len(test_df)
	mode_acc_dev = sum([1 if res == mode else 0 for res in dev_df.iloc[:][0]]) / len(dev_df)

	sb.append_line("Mode Label Accuracy on Test Set: %f" % mode_acc_test, True)
	sb.append_line("Mode Label Accuracy on Dev Set: %f" % mode_acc_dev, True)

	with open("mode_label.txt", "w") as f: sb.dump_output(f)

	exec_experiments(StandardPerceptron(), [1, 0.1, 0.01],
		train_df, test_df, dev_df, sb)
	exec_experiments(DecayPerceptron(), [1, 0.1, 0.01],
		train_df, test_df, dev_df, sb)
	exec_experiments(MarginPerceptron(), [1, 0.1, 0.01],
		train_df, test_df, dev_df, sb,
		[1, 0.1, 0.01])
	exec_experiments(AveragePerceptron(), [1, 0.1, 0.01],
		train_df, test_df, dev_df, sb)

if __name__ == "__main__": main()