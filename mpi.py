import sys

from mpi4py import MPI
from random import randint
from io import open
from pandas import DataFrame, set_option

comm = MPI.COMM_WORLD
worldSize = comm.Get_size()
rank = comm.Get_rank()

MATRIX_COUNT = 5
MATRIX_FACTOR = 4
MATRIX_SIZE = (worldSize - 1) * MATRIX_FACTOR

set_option('display.max_rows', MATRIX_SIZE)
set_option('display.max_columns', MATRIX_SIZE)
set_option('display.expand_frame_repr', False)


def checksum(m1, m2, res):
    m3 = generate_empty_result_matrix()
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            for k in range(len(m2)):
                m3[i][j] += m1[i][k] * m2[k][j]
    cksm1 = 0
    cksm2 = 0

    for i in res:
        for j in i:
            cksm1 += j

    for i in m3:
        for j in i:
            cksm2 += j

    return cksm1 == cksm2


def log(msg):
    if rank == 0:
        name = "MASTER"
    else:
        name = "SLAVE-{0}".format(rank - 1)
    with open('./LOGS/LOG-{0}.txt'.format(name), 'a+t') as stream:
        stream.write(msg)
        stream.write("\n")


def is_master():
    return rank == 0


def columns(mat):
    col_count = len(mat[0])
    cols = []
    for i in range(col_count):
        cols.append([row[i] for row in mat])
    return cols


def generate_random_matrix():
    return [[randint(1, 99) for _ in range(MATRIX_SIZE)] for _ in range(MATRIX_SIZE)]


def generate_empty_result_matrix():
    return [[0 for _ in range(MATRIX_SIZE)] for _ in range(MATRIX_SIZE)]


def distribute_matrices():
    matrix = generate_random_matrix()
    queue = [generate_random_matrix() for _ in range(MATRIX_COUNT - 1)]

    while len(queue) > 0:
        mat2 = queue.pop()
        log("Multiplying:\n\nMatrix 1:\n{0}\n\nMatrix 2:\n{1}".format(str(DataFrame(matrix)), str(DataFrame(mat2))))

        result = generate_empty_result_matrix()

        for i in range(1, worldSize):
            start_row = (i - 1) * MATRIX_FACTOR
            end_row = i * MATRIX_FACTOR
            rows = matrix[start_row:end_row]

            comm.send(rows, dest=i, tag=0)
            comm.send(mat2, dest=i, tag=1)

        for i in range(1, worldSize):
            worker_res = comm.recv(source=i, tag=i)
            skip = (i - 1) * MATRIX_FACTOR
            for j in range(MATRIX_FACTOR):
                for k in range(len(worker_res[0])):
                    result[skip + j][k] = worker_res[j][k]

        log("Checksum result: {0}\n".format(checksum(matrix, mat2, result)))

        matrix = result

    log("Final Result:\n{0}".format(str(DataFrame(matrix))))
    return matrix


def order_matrix(matrix):
    submtx = []
    for i in range(0, len(matrix), MATRIX_FACTOR):
        submtx.append(matrix[i:i + MATRIX_FACTOR])

    for i in range(1, worldSize):
        index = i-1
        comm.send(submtx[index], dest=i, tag=0)

    result = []

    for i in range(1, worldSize):
        res = comm.recv(source=i, tag=i)
        result += res

    result.sort()

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = result[j + i*len(matrix[0])]

    log("Final ordered matrix:\n{0}".format(str(DataFrame(matrix))))


def master_op():
    final_mat = distribute_matrices()
    order_matrix(final_mat)


def multiply_matrices():
    for _ in range(MATRIX_COUNT - 1):
        rows_x = comm.recv(source=0, tag=0)
        maty = comm.recv(source=0, tag=1)

        res = []

        cols = columns(maty)

        for row in rows_x:
            row_res = []
            for i in range(len(cols[0])):
                val = 0
                for j in range(len(row)):
                    val += row[j] * cols[i][j]
                row_res.append(val)
            res.append(row_res)

        log("Received:\nROWS:\n{0}\nMAT_Y:\n{1}\nCalculated:\n{2}\n".format(str(DataFrame(rows_x)), str(DataFrame(maty)),
                                                                          str(DataFrame(res))))

        comm.send(res, dest=0, tag=rank)


def order_submatrices():
    arrs = comm.recv(source=0, tag=0)

    log("Received:\n{0}\n".format(str(DataFrame(arrs))))

    merged = []
    for arr in arrs:
        merged += arr

    merged.sort()

    # matrix = []
    # for i in range(0, len(merged), line_len):
    #     matrix.append(merged[i:i + line_len])

    log("Sorted:\n{0}\n".format(str(merged)))
    comm.send(merged, dest=0, tag=rank)


def slave_op():
    multiply_matrices()
    order_submatrices()


if is_master():
    master_op()
else:
    slave_op()
