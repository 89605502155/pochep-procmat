# pochep-procmat

Матричные разложения для модели потребления памяти процессов `M t = p`.

Строки `M` соответствуют моментам времени, столбцы процессам. `p` это суммарная измеренная память
в каждый момент, `t` вектор весов процессов. Пакет решает эту линейную систему через QR, SVD,
нормальные уравнения с разложением Холецкого и неотрицательный МНК (NNLS), а также строит
неотрицательное разложение `M ~ W H` (NMF). Есть генератор синтетических профилей процессов для тестов.

Английская версия: [README.md](README.md). Документация: [docs/ru](docs/ru/theory.md), [docs/en](docs/en/theory.md).
Демонстрационный ноутбук с сильными и слабыми сторонами: https://github.com/89605502155/pochep-procmat-demo

## Установка

```bash
pip install pochep-procmat
```

Требуется Python 3.11–3.14 и NumPy 2.0 или новее. Страница на PyPI: https://pypi.org/project/pochep-procmat/

## Пример

```python
from pochep_procmat import ProcessMemoryModel, make_process_profiles

M, t_true, p = make_process_profiles(n_times=120, n_processes=4, seed=1, noise=0.01)
model = ProcessMemoryModel("svd")
result = model.fit(M, p)
print(result.weights, result.relative_residual, result.rank)
```

## Лицензия

MIT. Авторы: Ферубко Андрей Олегович, Казаков Олег Дмитриевич. Как цитировать: см. `CITATION.cff`.
