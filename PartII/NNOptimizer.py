from sklearn.metrics import accuracy_score
import mlrose
import time

def NN_RC_maximizer(max_attempts_range, restarts_range, X_train, X_test, y_train, y_test, threshold):
	result = {}
	result['threshold_reached'] = False
	result['train_accuracy'] = -1
	for max_attempts in max_attempts_range:
		for restarts in restarts_range:
			NN_RC = mlrose.NeuralNetwork(hidden_nodes = [10], activation = 'sigmoid',
									algorithm = 'random_hill_climb', max_iters = 10000,
									bias = True, is_classifier = True, learning_rate = 0.01,
									early_stopping = True, clip_max = 5, 
									max_attempts = max_attempts, restarts = restarts)
			t = time.time()
			NN_RC.fit(X_train, y_train)
			t = time.time()- t
			y_train_pred = NN_RC.predict(X_train)
			train_accuracy = accuracy_score(y_train, y_train_pred)
			if result['train_accuracy'] < train_accuracy:
				y_test_pred = NN_RC.predict(X_test)
				test_accuracy = accuracy_score(y_test, y_test_pred)
				result['train_accuracy'] = train_accuracy
				result['test_accuracy'] = test_accuracy
				result['train_time'] = t
				result['parameters'] = {'max_attempts': max_attempts,
                						'restarts': restarts}
			if train_accuracy >= threshold:
				result['threshold_reached'] = True
				return result

	return result


def NN_SA_maximizer(max_attempts_range, init_temp_range, decay, min_temp, X_train, X_test, y_train, y_test, threshold):
	result = {}
	result['threshold_reached'] = False
	result['train_accuracy'] = -1
	for max_attempts in max_attempts_range:
		for init_temp in init_temp_range:
			schedule = mlrose.GeomDecay(init_temp=init_temp, decay=decay, min_temp=min_temp)
			NN_SA = mlrose.NeuralNetwork(hidden_nodes = [10], schedule=schedule,
									algorithm = 'simulated_annealing', max_iters = 10000,
									bias = True, is_classifier = True, learning_rate = 0.01,
									early_stopping = True, clip_max = 5, 
									max_attempts = max_attempts)
			t = time.time()
			NN_SA.fit(X_train, y_train)
			t = time.time()- t
			y_train_pred = NN_SA.predict(X_train)
			train_accuracy = accuracy_score(y_train, y_train_pred)
			if result['train_accuracy'] < train_accuracy:
				y_test_pred = NN_SA.predict(X_test)
				test_accuracy = accuracy_score(y_test, y_test_pred)
				result['train_accuracy'] = train_accuracy
				result['test_accuracy'] = test_accuracy
				result['train_time'] = t
				result['parameters'] = {'max_attempts': max_attempts,
                						'init_temp': init_temp}
			if train_accuracy >= threshold:
				result['threshold_reached'] = True
				return result

	return result

def NN_GA_maximizer(max_attempts_range, pop_size_range, mutation_prob_range, X_train, X_test, y_train, y_test, threshold):
	result = {}
	result['threshold_reached'] = False
	result['train_accuracy'] = -1
	for max_attempts in max_attempts_range:
		for pop_size in pop_size_range:
			for mutation_prob in mutation_prob_range:
				NN_GA = mlrose.NeuralNetwork(hidden_nodes = [10], pop_size = pop_size, mutation_prob = mutation_prob,
										algorithm = 'genetic_alg', max_iters = 10000,
										bias = True, is_classifier = True, learning_rate = 0.01,
										early_stopping = True, clip_max = 5, 
										max_attempts = max_attempts)
				t = time.time()
				NN_GA.fit(X_train, y_train)
				t = time.time()- t
				y_train_pred = NN_GA.predict(X_train)
				train_accuracy = accuracy_score(y_train, y_train_pred)
				if result['train_accuracy'] < train_accuracy:
					y_test_pred = NN_GA.predict(X_test)
					test_accuracy = accuracy_score(y_test, y_test_pred)
					result['train_accuracy'] = train_accuracy
					result['test_accuracy'] = test_accuracy
					result['train_time'] = t
					result['parameters'] = {'max_attempts': max_attempts,
	                						'pop_size': pop_size,
	                						'mutation_prob': mutation_prob}
				if train_accuracy >= threshold:
					result['threshold_reached'] = True
					return result

	return result


def NN_RC_report(max_attempts_range, restarts_range, X_train, X_test, y_train, y_test):
	results = []
	for max_attempts in max_attempts_range:
		for restarts in restarts_range:
			NN_RC = mlrose.NeuralNetwork(hidden_nodes = [10], activation = 'sigmoid',
									algorithm = 'random_hill_climb', max_iters = 10000,
									bias = True, is_classifier = True, learning_rate = 0.01,
									early_stopping = True, clip_max = 5, 
									max_attempts = max_attempts, restarts = restarts)
			t = time.time()
			NN_RC.fit(X_train, y_train)
			t = time.time()- t
			y_train_pred = NN_RC.predict(X_train)
			train_accuracy = accuracy_score(y_train, y_train_pred)
			y_test_pred = NN_RC.predict(X_test)
			test_accuracy = accuracy_score(y_test, y_test_pred)
			result = {}
			result['max_attempts'] = max_attempts
			result['restarts'] = restarts
			result['train_accuracy'] = train_accuracy
			result['test_accuracy'] = test_accuracy
			result['train_time'] = t
			results.append(result)

	return results

def NN_SA_report(max_attempts_range, init_temp_range, decay, min_temp_range, X_train, X_test, y_train, y_test):
	results = []
	for max_attempts in max_attempts_range:
		for init_temp in init_temp_range:
			for min_temp in min_temp_range:
				schedule = mlrose.GeomDecay(init_temp=init_temp, decay=decay, min_temp=min_temp)
				NN_SA = mlrose.NeuralNetwork(hidden_nodes = [10], schedule=schedule,
										algorithm = 'simulated_annealing', max_iters = 10000,
										bias = True, is_classifier = True, learning_rate = 0.01,
										early_stopping = True, clip_max = 5, 
										max_attempts = max_attempts)
				t = time.time()
				NN_SA.fit(X_train, y_train)
				t = time.time()- t
				y_train_pred = NN_SA.predict(X_train)
				train_accuracy = accuracy_score(y_train, y_train_pred)
				y_test_pred = NN_SA.predict(X_test)
				test_accuracy = accuracy_score(y_test, y_test_pred)
				result = {}
				result['max_attempts'] = max_attempts
				result['init_temp'] = init_temp
				result['min_temp'] = min_temp			
				result['train_accuracy'] = train_accuracy
				result['test_accuracy'] = test_accuracy
				result['train_time'] = t
				results.append(result)

	return results


def NN_GA_report(max_attempts_range, pop_size_range, mutation_prob_range, X_train, X_test, y_train, y_test):
	results = []
	for max_attempts in max_attempts_range:
		for pop_size in pop_size_range:
			for mutation_prob in mutation_prob_range:
				NN_GA = mlrose.NeuralNetwork(hidden_nodes = [10], pop_size = pop_size, mutation_prob = mutation_prob,
										algorithm = 'genetic_alg', max_iters = 10000,
										bias = True, is_classifier = True, learning_rate = 0.01,
										early_stopping = True, clip_max = 5, 
										max_attempts = max_attempts)
				t = time.time()
				NN_GA.fit(X_train, y_train)
				t = time.time()- t
				y_train_pred = NN_GA.predict(X_train)
				train_accuracy = accuracy_score(y_train, y_train_pred)
				y_test_pred = NN_GA.predict(X_test)
				test_accuracy = accuracy_score(y_test, y_test_pred)
				result = {}
				result['max_attempts'] = max_attempts
				result['pop_size'] = pop_size
				result['mutation_prob'] = mutation_prob
				result['train_accuracy'] = train_accuracy
				result['test_accuracy'] = test_accuracy
				result['train_time'] = t
				results.append(result)

	return results
