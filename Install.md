# Установка

## Windows / Linux / macOS

Требуется Python 3.10 или новее.

```powershell
python -m pip install -e ".[test]"
python -m pytest
```

## Зависимости

- NumPy — векторы, плотные якобианы, SVD/lstsq
- SciPy — разреженный Ньютон (`splu`), факторизация Холецкого для Goldfarb–Idnani
- pytest — тесты

Исходный C++-проект использует Eigen3, GTest и eiquadprog. В Python-порте линейная алгебра — NumPy/SciPy, QP Goldfarb–Idnani перенесён в `fixed_solvers/qp/eiquadprog.py`.

Пайплайн: `.github/workflows/tests.yml`, `.gitlab-ci.yml`, `pipeline/pipeline_scripts/`.
