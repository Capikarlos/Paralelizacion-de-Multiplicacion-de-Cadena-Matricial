import os

def fijar_un_hilo_blas():
    """
    Debe invocarse antes de operaciones pesadas para medir el Speedup
    de forma justa entre el motor secuencial (1 hilo) y los motores paralelos.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")