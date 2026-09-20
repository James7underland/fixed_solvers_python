"""Реэкспорт QP: Goldfarb–Idnani и CCS-представление разреженных матриц."""
from .eiquadprog import (
    add_constraint,
    compute_d,
    delete_constraint,
    distance,
    solve_quadprog,
    solve_quadprog2,
    update_r,
    update_z,
)
from .qp_wrapper import get_sparse_matrix_CCS, solve_quadprog_box

__all__ = [
    "add_constraint",
    "compute_d",
    "delete_constraint",
    "distance",
    "get_sparse_matrix_CCS",
    "solve_quadprog",
    "solve_quadprog2",
    "solve_quadprog_box",
    "update_r",
    "update_z",
]
