from mpi4py import MPI
from random import randint
from io import open
from pandas import DataFrame

comm = MPI.COMM_WORLD
worldSize = comm.Get_size()
rank = comm.Get_rank()

MATRIX_FACTOR = 100
MATRIX_SIZE = (worldSize - 1) * MATRIX_FACTOR


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


def master_op():
    mtx1 = generate_random_matrix()
    mtx2 = generate_random_matrix()
    mtx3 = generate_empty_result_matrix()

    for i in range(1, worldSize):
        start_row = (i - 1) * MATRIX_FACTOR
        end_row = i * MATRIX_FACTOR
        rows = mtx1[start_row:end_row]

        comm.send(rows, dest=i, tag=0)
        comm.send(mtx2, dest=i, tag=1)
        pass

    for i in range(1, worldSize):
        worker_res = comm.recv(source=i, tag=i)
        skip = (i - 1) * MATRIX_FACTOR
        for j in range(MATRIX_FACTOR):
            for k in range(len(worker_res[0])):
                mtx3[skip + j][k] = worker_res[j][k]

    log("Matrix 1:\n{0}\nMatrix 2:\n{1}\nResult:\n{2}".format(str(DataFrame(mtx1)), str(DataFrame(mtx2)),
                                                              str(DataFrame(mtx3))))


def slave_op():
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

    log("Received:\nROWS:\n{0}\nMAT_Y:\n{1}\nCalculated:\n{2}".format(str(DataFrame(rows_x)), str(DataFrame(maty)),
                                                                      str(DataFrame(res))))

    comm.send(res, dest=0, tag=rank)


if is_master():
    master_op()
else:
    slave_op()
