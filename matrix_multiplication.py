# Program to multiply two matrices using nested loops

def matrix_multiplication(X, Y):
    x_rows = len(X)
    x_columns = len(X[0])
    y_rows = len(Y)
    y_columns = len(Y[0])

    # check to see if the matrices can be multiplied
    # i.e. x_column must equal y_rows

    if x_columns != y_rows:
        raise Exception("This matrix multiplication is not possible.")

    #create an empty list of lists to initialize matrix to all zeros
    result = [[0 for i in range(y_columns)] for j in range(x_rows)]

    # iterate through rows of X
    for i in range(len(X)):
       # iterate through columns of Y
        for j in range(len(Y[0])):
           # iterate through rows of Y
            for k in range(len(Y)):
                result[i][j] += X[i][k] * Y[k][j]

    return result


if __name__=="__main__":
    # 3x3 matrix
    X = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]

    # 3x4 matrix
    Y = [[5, 8, 1, 2],
         [6, 7, 3, 0],
         [4, 5, 9, 1]]

    # result is 3x4
    # result = [[0, 0, 0, 0],
    #           [0, 0, 0, 0],
    #           [0, 0, 0, 0]]

    print(matrix_multiplication(X,Y))
