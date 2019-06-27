#This is a simple Perception algorithm

def predict(inputs, weights):
    """
    :param inputs: <list>
    :param weights: <list>
    :return: boolean
    """
    # This is the prediction function
    # multiply the weights by the features and sum them up
    # e.g. i0 x w0 + i1 x w1 + i2 x w2

    # the first element in the list of the weight is the bias factor
    activation = weights[0]

    # remove bias value from weights before calculating the total activation
    weights_without_bias = weights[1:]

    for i, w in zip(inputs, weights_without_bias):
        activation += i * w
    if activation >= 0.0:
        return 1.0
    else:
        return 0.0


def train_weights(feature, learning_rate, n_epoch):
    """
    function to estimate Perception weights
    :param feature: <list> input data
    :param learning_rate: <float>
    :param n_epoch: <int>
    :return: weights: <list>
    """

    # init a list of n weights to be zero
    weights = [0.0 for i in range(len(feature[0]))]

    for epoch in range(n_epoch):
        sum_error = 0
        for row in feature:
            prediction = predict(row[:-1], weights)
            error = row[-1] - prediction
            sum_error += error**2

            # the bias part of the weight is updated but without an input
            weights[0] = weights[0] + learning_rate*error

            # calculate the new weights
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + learning_rate * error * row[i]

        print('>>epoch=%d, learn_rate=%.3f, error=%.3f' % (epoch, learning_rate, sum_error))

    return weights


if __name__ == "__main__":
    #           [feature1, feature2, label]
    dataset = [[2.7810836, 2.550537003, 0],
               [1.465489372, 2.362125076, 0],
               [3.396561688, 4.400293529, 0],
               [1.38807019, 1.850220317, 0],
               [3.06407232, 3.005305973, 0],
               [7.627531214, 2.759262235, 1],
               [5.332441248, 2.088626775, 1],
               [6.922596716, 1.77106367, 1],
               [8.675418651, -0.242068655, 1],
               [7.673756466, 3.508563011, 1]]

    label = [dataset[i][-1] for i in range(len(dataset))]
    features = [dataset[i][0:-1] for i in range(len(dataset))]

    # weights = [-0.1, 0.20653640140000007, -0.23418117710000003]

    l_rate = 0.1
    n_epoch = 5
    weights = train_weights(dataset, l_rate, n_epoch)
    print('weights:', weights)
