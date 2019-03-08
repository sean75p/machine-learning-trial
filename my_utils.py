class MyDataCalc:
    def __init__(self, value):
        # value should be in list format
        self.value = value
        self.intermediate_list = []
        self.mean = sum(self.value) / len(self.value)

        for i in self.value:
            self.intermediate_list.append(((i - self.mean) ** 2))

        self.sigma_squared = sum(self.intermediate_list) / len(self.intermediate_list)
        self.sigma = (self.sigma_squared) ** 0.5
        self.standard_deviation = self.sigma

    def mean(self):
        """
        :return: mean
        """
        return self.mean

    def standardize(self):
        # standardize data, subtract mean from each feature and divide by the standard deviation
        """
        :param value: list
        :return: standardized value: list
        """
        standardized_data = []
        for i in self.value:
            standardized_data.append((i - self.mean)/self.sigma)

        return standardized_data

if __name__ == '__main__':
    a = [1, 2, 3, 4, 6]
    instance = MyDataCalc(a)
    print('mean:', instance.mean)
    print('standard deviation:', instance.sigma)
    b = instance.standardize()
    print('standardized list:', b)
    instance = MyDataCalc(b)
    print('the mean of a standardized list should be 0:', instance.mean )
    print('the std deviation of a standardized list should be 1:', instance.standard_deviation)