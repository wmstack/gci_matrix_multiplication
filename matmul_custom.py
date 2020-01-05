import numba

@numba.jit
def dot_loop_jit(x,y):
    sum = 0
    for i in range(len(x)):
        sum+= x[i]*y[i]
    return sum

@numba.jit
def matmul_jit(x, y):
    #generic matmul
    #not just matmul of n x n matrices


    #matmul, the length of a row of the first matrix 'x'
    #is equal to the length of a column of the second matrix 'y'
    #so we do the dot_product of the ith row in x, and jth column in y
    #then put the result in (i,j) in the resulting array
    
    #note that the shape is (rows, cells)
    #so the length of the row is the number of cells
    #the length of a column is the number of rows

    row_len_x = len(x[0])
    col_len_y = len(y)

    assert row_len_x == col_len_y
    
    #resulting dimensions
    row_len_y = len(y[0])
    col_len_x = len(x)
    

    # a dot product of the ith row in x, x[i]
    # and jth col in y,  [y[k][j] for k in range(col_len_y)])
    # for each (i, j)
    # or . for j in range(row_len_y)
    # for i in range(col_len_x)

    #the for j comes first becuase shape is (rows, cells)
    #and the cells of the row are the inner arrays so loop through the cells first
    #then loop through the rows
    
    #otherwise it is transposed

    return [[dot_loop_jit(x[i],[y[k][j] for k in range(col_len_y)]) for j in range(row_len_y)] for i in range(col_len_x)]
    #return [[dot_loop(x[i],y[:,j]) for j in range(row_len_y)] for i in range(col_len_x)]


def dot_loop_no_jit(x,y):
    sum = 0
    for i in range(len(x)):
        sum+= x[i]*y[i]
    return sum


def matmul_no_jit(x, y):
    #generic matmul
    #not just matmul of n x n matrices


    #matmul, the length of a row of the first matrix 'x'
    #is equal to the length of a column of the second matrix 'y'
    #so we do the dot_product of the ith row in x, and jth column in y
    #then put the result in (i,j) in the resulting array
    
    #note that the shape is (rows, cells)
    #so the length of the row is the number of cells
    #the length of a column is the number of rows

    row_len_x = len(x[0])
    col_len_y = len(y)

    assert row_len_x == col_len_y
    
    #resulting dimensions
    row_len_y = len(y[0])
    col_len_x = len(x)
    

    # a dot product of the ith row in x, x[i]
    # and jth col in y,  [y[k][j] for k in range(col_len_y)])
    # for each (i, j)
    # or . for j in range(row_len_y)
    # for i in range(col_len_x)

    #the for j comes first becuase shape is (rows, cells)
    #and the cells of the row are the inner arrays so loop through the cells first
    #then loop through the rows
    
    #otherwise it is transposed

    return [[dot_loop_no_jit(x[i],[y[k][j] for k in range(col_len_y)]) for j in range(row_len_y)] for i in range(col_len_x)]
    #return [[dot_loop(x[i],y[:,j]) for j in range(row_len_y)] for i in range(col_len_x)]