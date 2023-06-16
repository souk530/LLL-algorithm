import time 
def generate_matrix(n, m, low=1, high=10):
    matrix = random_matrix(ZZ, n, x = low, y = high)
    return matrix % m

def is_linearly_independent(vectors):
    matrix = Matrix(vectors)
    if matrix.rank() == matrix.nrows():
        return True
    else:
        return False

num = 30
m = 37

while True:
    vectors = generate_matrix(num, m)
    print("original:\n", vectors)
    if is_linearly_independent(vectors) == False:
        print("線形従属")
        continue
    else:
        matrix_LLL = vectors.LLL()
        print("after LLL:\n", matrix_LLL % m)
        time.sleep(3)
