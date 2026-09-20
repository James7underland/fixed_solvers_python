# Решатели фиксированной размерности (Python)

Библиотека `fixed_solvers`: метод Ньютона–Рафсона для систем нелинейных уравнений известной или переменной размерности, бисекция/секущие, оптимизация Гаусса–Ньютона, линейный поиск и вспомогательная математика.

Векторы и матрицы хранятся в NumPy; разреженные Якобианы переменной размерности решаются через SciPy.

## Установка

```powershell
python -m pip install -e ".[test]"
```

Зависимости: `numpy`, `scipy`. Для тестов: `pytest`.

## Учебный пример

Система

```math
\begin{cases}
(x_{1} - 2)^{3} = 8 \\
(x_{2} - 1)^{3} = 27
\end{cases}
```

в виде невязок \(r(x)=0\):

```python
import numpy as np
from fixed_solvers import (
    fixed_newton_raphson,
    fixed_solver_parameters_t,
    fixed_solver_result_t,
    fixed_system_t,
)

class sample_system(fixed_system_t):
    dimension = 2

    def residuals(self, x):
        x = np.asarray(x, dtype=float)
        return np.array(
            [
                (x[0] - 2.0) ** 3 - 8.0,
                (x[1] - 1.0) ** 3 - 27.0,
            ]
        )

system = sample_system()
parameters = fixed_solver_parameters_t(2, 0)
result = fixed_solver_result_t(2)
fixed_newton_raphson[2].solve_dense(system, np.array([0.0, 0.0]), parameters, result)
print(result.result_code, result.argument)
```

Аналитический якобиан можно переопределить методом `jacobian_dense` (для `dimension == -1` — `jacobian_sparse`). По умолчанию используется двусторонняя численная производная.

## Состав пакета

```
fixed_solvers/
  fixed_system.py              — системы уравнений, ц.ф., численный якобиан
  fixed_linear_solver.py       — СЛАУ 1×1 / 2×2 / 3×3 (Крамер)
  fixed_constraints.py         — box- и линейные ограничения
  fixed_nonlinear_solver.py    — Ньютон–Рафсон
  fixed_optimizer.py           — Гаусс–Ньютон (сумма квадратов)
  fixed_bisection.py           — бисекция / секущие / Illinois
  line_search/                 — дробление шага и золотое сечение (в т.ч. ООФ)
  helpers/                     — кубические уравнения, кусочные полиномы, math/string
  array_ext.py                 — операции над массивами фиксированной размерности
  qp/                          — Goldfarb–Idnani QP (eiquadprog), box-QP и CCS
```

## Параметры и результат Ньютона–Рафсона

| Объект | Назначение |
| --- | --- |
| `fixed_solver_parameters_t` | итерации, нормы, line search, ограничения, диагностика |
| `fixed_solver_result_t` | код, балл сходимости, аргумент, невязки |
| `fixed_solver_result_analysis_t` | история аргумента, шагов и исследования ц.ф. |
| `fixed_solver_constraints` | min / max / relative |

Линейный поиск: `divider_search` (по умолчанию), `golden_section_search`, `golden_section_search_domain_discovery`, `no_line_search`.

Выход за область определения функции сигнализируется исключением `domain_violation` (наследует `BaseException`, не `Exception`).

Квадратичное программирование — dual-метод Goldfarb–Idnani (`solve_quadprog` / `solve_quadprog_box`).

## Тесты

```powershell
python -m pytest
```

Тесты покрывают солверы, QP, `array_ext`, sparse-ограничения и строковые хелперы.

## CI/CD

GitHub Actions в репозитории:

- **CI** (`ci.yml`) на `push`/`pull_request` в `main`: ruff, pytest на Python 3.10–3.12 (Linux) и 3.12 (Windows), сборка sdist/wheel.
- **Release** (`release.yml`): тег вида `v1.0.1` собирает пакет, прогоняет тесты и публикует GitHub Release с артефактами.
- **Dependabot** раз в неделю обновляет Actions и pip-зависимости.

Локально:

```powershell
python -m pip install -e ".[test,dev]"
python -m ruff check .
python -m pytest --cov=fixed_solvers
python -m build
```

## Документация алгоритмов

Исходные описания ООФ и порога шума line search:

- `documents/feat-function-domain.md`
- `documents/feat-linesearch-noise-floor.md`
- `documents/linesearch-noise-floor-research-paper.md`
