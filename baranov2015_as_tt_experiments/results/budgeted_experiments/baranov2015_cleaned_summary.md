# Baranov 2015 / water PES: cleaned comparison report

Этот файл — аккуратная версия исходного отчёта. Главная правка: вместо 15 строк `my_cheb_r*_lam*` для каждого `(n, budget)` оставлена только одна строка **Cheb-CP best**, выбранная по минимальному `off-grid RMSE (meV)`.

## Что сравнивается

| Параметр | Значение |
|---|---|
| Статус | complete |
| Последнее обновление | 2026-05-01 08:32:16 MSK |
| Молекула/геометрия | `water.xyz` |
| Единицы координат | Bohr |
| Active subspace dim | 4 |
| σ² для AS | 0.1 |
| AS samples | 512 |
| Chebyshev interval | [-0.3, 0.3] |
| Chebyshev grids | [2, 3, 4, 5, 6, 7, 8, 9, 10] |
| Run mode | budget_sweep |
| Sampling | fraction |
| Authors backend | octave_tt_toolbox |

## Протокол сравнения

- Все fixed-mask completion baselines получают один и тот же префикс author-selected trace: одинаковые observed points, одинаковую mask и одинаковые значения PES.
- Основная метрика — **off-grid RMSE (meV)** на случайных точках AS-домена.
- `hidden RMSE` считается только для completion rows, где есть скрытые grid entries.
- `AS+TT authors` — отдельный paper-style baseline через adaptive TT-cross. Его нельзя напрямую считать fixed-mask completion методом, потому что он сам выбирает точки.

## Легенда методов

| Метод | Что показано |
|---|---|
| Cheb-CP best | лучший вариант `my_cheb_r*_lam*` для данного `(n, budget)` по off-grid RMSE |
| CP-WOPT (Tensor Toolbox) | внешний CP-WOPT baseline, rank 12 |
| RTTC-TT (TTeMPS) | внешний TT-completion baseline, rank 12 |
| Tucker (TensorLy) | masked Tucker baseline, rank 12 |
| HaLRTC (XinyChen) | external HaLRTC/LRTC baseline |
| AS+TT authors | paper-style Baranov/Oseledets через Octave + TT-Toolbox `dmrg_cross`; не fixed-mask completion |

## Основной результат на максимальном доступном бюджете

Эта таблица показывает наиболее важную часть эксперимента: для каждого `n` взят максимальный доступный budget, а Cheb-CP оставлен только в лучшей конфигурации rank/λ.

| n | budget | method | rank/param | evals | off-grid RMSE | off-grid max | hidden RMSE | time(s) |
|---|---|---|---|---|---|---|---|---|
| 2 | 16 | Cheb-CP best | r=50, λ=1 | 20 | 410.9 | 1 015.2 | n/a | 7.852 |
| 2 | 16 | CP-WOPT (Tensor Toolbox) | 12 | 20 | 410.9 | 1 016.1 | n/a | 3.032 |
| 2 | 16 | HaLRTC (XinyChen) | n/a | 20 | 410.9 | 1 016.1 | n/a | 0.000 |
| 2 | 16 | RTTC-TT (TTeMPS) | 12 | 20 | 410.9 | 1 016.1 | n/a | 3.106 |
| 2 | 16 | Tucker (TensorLy) | 12 | 20 | 410.9 | 1 016.1 | n/a | 0.001 |
| 3 | 81 | Cheb-CP best | r=50, λ=1 | 240 | 65.01 | 143.9 | n/a | 3.466 |
| 3 | 81 | CP-WOPT (Tensor Toolbox) | 12 | 240 | 65.04 | 143.9 | n/a | 3.183 |
| 3 | 81 | HaLRTC (XinyChen) | n/a | 240 | 65.04 | 143.9 | n/a | 0.000 |
| 3 | 81 | RTTC-TT (TTeMPS) | 12 | 240 | 65.04 | 143.9 | n/a | 3.097 |
| 3 | 81 | Tucker (TensorLy) | 12 | 240 | 65.04 | 143.9 | n/a | 0.001 |
| 4 | 256 | Cheb-CP best | r=50, λ=0.1 | 575 | 7.515 | 18.10 | n/a | 2.933 |
| 4 | 256 | CP-WOPT (Tensor Toolbox) | 12 | 575 | 7.517 | 17.92 | n/a | 3.199 |
| 4 | 256 | HaLRTC (XinyChen) | n/a | 575 | 7.517 | 17.92 | n/a | 0.000 |
| 4 | 256 | RTTC-TT (TTeMPS) | 12 | 575 | 7.517 | 17.92 | n/a | 3.061 |
| 4 | 256 | Tucker (TensorLy) | 12 | 575 | 7.517 | 17.92 | n/a | 0.001 |
| 5 | 625 | Cheb-CP best | r=100, λ=0 | 1049 | 0.759 | 1.827 | n/a | 4.210 |
| 5 | 625 | CP-WOPT (Tensor Toolbox) | 12 | 1049 | 0.759 | 1.827 | n/a | 3.259 |
| 5 | 625 | HaLRTC (XinyChen) | n/a | 1049 | 0.759 | 1.827 | n/a | 0.000 |
| 5 | 625 | RTTC-TT (TTeMPS) | 12 | 1049 | 0.759 | 1.827 | n/a | 2.878 |
| 5 | 625 | Tucker (TensorLy) | 12 | 1049 | 0.759 | 1.827 | n/a | 0.002 |
| 6 | 1276 | Cheb-CP best | r=50, λ=0 | 3098 | 0.072 | 0.200 | 0.097 | 5.601 |
| 6 | 1276 | HaLRTC (XinyChen) | n/a | 3098 | 0.114 | 0.935 | 0.614 | 0.006 |
| 6 | 1276 | CP-WOPT (Tensor Toolbox) | 12 | 3098 | 4.085 | 37.04 | 31.75 | 3.099 |
| 6 | 1276 | RTTC-TT (TTeMPS) | 12 | 3098 | 25.77 | 236.9 | 190.0 | 3.395 |
| 6 | 1276 | Tucker (TensorLy) | 12 | 3098 | 45.94 | 456.8 | 373.5 | 0.002 |
| 7 | 2114 | Cheb-CP best | r=50, λ=0 | 4962 | 0.417 | 3.690 | 1.861 | 7.225 |
| 7 | 2114 | CP-WOPT (Tensor Toolbox) | 12 | 4962 | 18.69 | 98.21 | 50.19 | 3.138 |
| 7 | 2114 | HaLRTC (XinyChen) | n/a | 4962 | 47.62 | 277.1 | 151.2 | 0.048 |
| 7 | 2114 | Tucker (TensorLy) | 12 | 4962 | 229.8 | 1 251.6 | 744.7 | 0.003 |
| 7 | 2114 | RTTC-TT (TTeMPS) | 12 | 4962 | 235.0 | 1 213.7 | 673.9 | 4.158 |
| 8 | 2640 | Cheb-CP best | r=30, λ=0 | 7804 | 1.148 | 6.761 | 1.711 | 3.943 |
| 8 | 2640 | CP-WOPT (Tensor Toolbox) | 12 | 7804 | 20.98 | 73.57 | 30.32 | 3.113 |
| 8 | 2640 | HaLRTC (XinyChen) | n/a | 7804 | 146.6 | 721.6 | 317.5 | 0.076 |
| 8 | 2640 | Tucker (TensorLy) | 12 | 7804 | 288.0 | 1 263.4 | 671.2 | 0.003 |
| 8 | 2640 | RTTC-TT (TTeMPS) | 12 | 7804 | 352.1 | 1 333.0 | 663.6 | 4.244 |

## Победители fixed-mask сравнения по каждому `(n, budget)`

Считаются только fixed-mask методы: Cheb-CP best, CP-WOPT, RTTC-TT, Tucker, HaLRTC. AS+TT authors не включён, потому что это adaptive TT-cross baseline.

| n | budget | winner | rank/param | off-grid RMSE |
|---|---|---|---|---|
| 2 | 4 | Cheb-CP best | r=5, λ=1 | 400.5 |
| 2 | 8 | Cheb-CP best | r=1, λ=1 | 394.5 |
| 2 | 16 | Cheb-CP best | r=50, λ=1 | 410.9 |
| 3 | 17 | CP-WOPT (Tensor Toolbox) | 12 | 393.8 |
| 3 | 41 | CP-WOPT (Tensor Toolbox) | 12 | 275.7 |
| 3 | 81 | Cheb-CP best | r=50, λ=1 | 65.01 |
| 4 | 52 | HaLRTC (XinyChen) | n/a | 363.0 |
| 4 | 128 | CP-WOPT (Tensor Toolbox) | 12 | 251.6 |
| 4 | 256 | Cheb-CP best | r=50, λ=0.1 | 7.515 |
| 5 | 125 | Cheb-CP best | r=1, λ=0 | 483.1 |
| 5 | 313 | CP-WOPT (Tensor Toolbox) | 12 | 67.66 |
| 5 | 625 | Cheb-CP best | r=100, λ=0 | 0.759 |
| 6 | 260 | CP-WOPT (Tensor Toolbox) | 12 | 475.3 |
| 6 | 648 | Cheb-CP best | r=5, λ=1 | 74.22 |
| 6 | 1276 | Cheb-CP best | r=50, λ=0 | 0.072 |
| 7 | 481 | Tucker (TensorLy) | 12 | 507.9 |
| 7 | 1201 | CP-WOPT (Tensor Toolbox) | 12 | 59.22 |
| 7 | 2114 | Cheb-CP best | r=50, λ=0 | 0.417 |
| 8 | 820 | CP-WOPT (Tensor Toolbox) | 12 | 265.0 |
| 8 | 2048 | CP-WOPT (Tensor Toolbox) | 12 | 56.56 |
| 8 | 2640 | Cheb-CP best | r=30, λ=0 | 1.148 |

Итого по всем 21 fixed-mask настройкам: Cheb-CP best: 11; CP-WOPT (Tensor Toolbox): 8; HaLRTC (XinyChen): 1; Tucker (TensorLy): 1.

## Лучшие Cheb-CP параметры

| n | budget | best Cheb | rank | λ | off-grid RMSE | off-grid max | hidden RMSE | time(s) |
|---|---|---|---|---|---|---|---|---|
| 2 | 4 | r5_lam1 | 5 | 1 | 400.5 | 994.7 | 89.94 | 0.613 |
| 2 | 8 | r1_lam1 | 1 | 1 | 394.5 | 1 003.9 | 132.6 | 0.240 |
| 2 | 16 | r50_lam1 | 50 | 1 | 410.9 | 1 015.2 | n/a | 7.852 |
| 3 | 17 | r100_lam0p1 | 100 | 0.1 | 430.8 | 1 635.7 | 856.0 | 19.09 |
| 3 | 41 | r30_lam0p1 | 30 | 0.1 | 299.6 | 1 415.1 | 776.9 | 1.662 |
| 3 | 81 | r50_lam1 | 50 | 1 | 65.01 | 143.9 | n/a | 3.466 |
| 4 | 52 | r100_lam1 | 100 | 1 | 390.1 | 2 145.6 | 762.5 | 19.95 |
| 4 | 128 | r50_lam1 | 50 | 1 | 258.8 | 1 324.6 | 783.6 | 4.500 |
| 4 | 256 | r50_lam0p1 | 50 | 0.1 | 7.515 | 18.10 | n/a | 2.933 |
| 5 | 125 | r1_lam0 | 1 | 0 | 483.1 | 1 991.7 | 824.2 | 0.868 |
| 5 | 313 | r30_lam1 | 30 | 1 | 152.4 | 979.5 | 331.8 | 1.871 |
| 5 | 625 | r100_lam0 | 100 | 0 | 0.759 | 1.827 | n/a | 4.210 |
| 6 | 260 | r5_lam1 | 5 | 1 | 493.6 | 1 645.5 | 751.4 | 0.600 |
| 6 | 648 | r5_lam1 | 5 | 1 | 74.22 | 216.5 | 77.30 | 0.677 |
| 6 | 1276 | r50_lam0 | 50 | 0 | 0.072 | 0.200 | 0.097 | 5.601 |
| 7 | 481 | r1_lam1 | 1 | 1 | 536.8 | 1 837.3 | 819.3 | 0.638 |
| 7 | 1201 | r5_lam0p1 | 5 | 0.1 | 74.71 | 345.5 | 127.3 | 0.720 |
| 7 | 2114 | r50_lam0 | 50 | 0 | 0.417 | 3.690 | 1.861 | 7.225 |
| 8 | 820 | r50_lam1 | 50 | 1 | 321.2 | 1 445.6 | 588.9 | 5.900 |
| 8 | 2048 | r30_lam0p1 | 30 | 0.1 | 69.79 | 307.1 | 115.3 | 3.746 |
| 8 | 2640 | r30_lam0 | 30 | 0 | 1.148 | 6.761 | 1.711 | 3.943 |

## Полный компактный budget sweep

Ниже оставлены все бюджеты, но для Cheb-CP в каждом бюджете показана только лучшая конфигурация rank/λ.

### n = 2

| budget | method | rank/param | evals | off-grid RMSE | hidden RMSE | time(s) |
|---|---|---|---|---|---|---|
| 4 | Cheb-CP best | r=5, λ=1 | 4 | 400.5 | 89.94 | 0.613 |
| 4 | CP-WOPT (Tensor Toolbox) | 12 | 4 | 400.8 | 92.22 | 2.835 |
| 4 | HaLRTC (XinyChen) | n/a | 4 | 401.0 | 89.94 | 0.000 |
| 4 | RTTC-TT (TTeMPS) | 12 | 4 | 401.1 | 89.89 | 3.047 |
| 4 | Tucker (TensorLy) | 12 | 4 | 401.4 | 89.56 | 0.001 |
| 8 | Cheb-CP best | r=1, λ=1 | 8 | 394.5 | 132.6 | 0.240 |
| 8 | HaLRTC (XinyChen) | n/a | 8 | 417.3 | 51.08 | 0.000 |
| 8 | RTTC-TT (TTeMPS) | 12 | 8 | 417.4 | 51.17 | 3.126 |
| 8 | Tucker (TensorLy) | 12 | 8 | 417.5 | 51.62 | 0.001 |
| 8 | CP-WOPT (Tensor Toolbox) | 12 | 8 | 435.0 | 96.01 | 2.951 |
| 16 | Cheb-CP best | r=50, λ=1 | 20 | 410.9 | n/a | 7.852 |
| 16 | CP-WOPT (Tensor Toolbox) | 12 | 20 | 410.9 | n/a | 3.032 |
| 16 | HaLRTC (XinyChen) | n/a | 20 | 410.9 | n/a | 0.000 |
| 16 | RTTC-TT (TTeMPS) | 12 | 20 | 410.9 | n/a | 3.106 |
| 16 | Tucker (TensorLy) | 12 | 20 | 410.9 | n/a | 0.001 |

### n = 3

| budget | method | rank/param | evals | off-grid RMSE | hidden RMSE | time(s) |
|---|---|---|---|---|---|---|
| 17 | CP-WOPT (Tensor Toolbox) | 12 | 17 | 393.8 | 779.0 | 3.133 |
| 17 | Cheb-CP best | r=100, λ=0.1 | 17 | 430.8 | 856.0 | 19.09 |
| 17 | HaLRTC (XinyChen) | n/a | 17 | 453.7 | 888.6 | 0.004 |
| 17 | Tucker (TensorLy) | 12 | 17 | 454.5 | 896.7 | 0.001 |
| 17 | RTTC-TT (TTeMPS) | 12 | 17 | 455.8 | 898.6 | 3.011 |
| 41 | CP-WOPT (Tensor Toolbox) | 12 | 50 | 275.7 | 728.3 | 3.254 |
| 41 | HaLRTC (XinyChen) | n/a | 50 | 292.4 | 762.3 | 0.005 |
| 41 | Cheb-CP best | r=30, λ=0.1 | 50 | 299.6 | 776.9 | 1.662 |
| 41 | Tucker (TensorLy) | 12 | 50 | 359.9 | 886.9 | 0.001 |
| 41 | RTTC-TT (TTeMPS) | 12 | 50 | 361.1 | 889.1 | 3.162 |
| 81 | Cheb-CP best | r=50, λ=1 | 240 | 65.01 | n/a | 3.466 |
| 81 | CP-WOPT (Tensor Toolbox) | 12 | 240 | 65.04 | n/a | 3.183 |
| 81 | HaLRTC (XinyChen) | n/a | 240 | 65.04 | n/a | 0.000 |
| 81 | RTTC-TT (TTeMPS) | 12 | 240 | 65.04 | n/a | 3.097 |
| 81 | Tucker (TensorLy) | 12 | 240 | 65.04 | n/a | 0.001 |

### n = 4

| budget | method | rank/param | evals | off-grid RMSE | hidden RMSE | time(s) |
|---|---|---|---|---|---|---|
| 52 | HaLRTC (XinyChen) | n/a | 68 | 363.0 | 716.4 | 0.010 |
| 52 | Tucker (TensorLy) | 12 | 68 | 378.8 | 721.0 | 0.002 |
| 52 | RTTC-TT (TTeMPS) | 12 | 68 | 386.6 | 728.0 | 2.871 |
| 52 | Cheb-CP best | r=100, λ=1 | 68 | 390.1 | 762.5 | 19.95 |
| 52 | CP-WOPT (Tensor Toolbox) | 12 | 68 | 483.2 | 865.5 | 2.920 |
| 128 | CP-WOPT (Tensor Toolbox) | 12 | 144 | 251.6 | 746.7 | 3.064 |
| 128 | Tucker (TensorLy) | 12 | 144 | 255.4 | 775.6 | 0.002 |
| 128 | RTTC-TT (TTeMPS) | 12 | 144 | 258.2 | 785.2 | 3.002 |
| 128 | Cheb-CP best | r=50, λ=1 | 144 | 258.8 | 783.6 | 4.500 |
| 128 | HaLRTC (XinyChen) | n/a | 144 | 258.8 | 783.6 | 0.000 |
| 256 | Cheb-CP best | r=50, λ=0.1 | 575 | 7.515 | n/a | 2.933 |
| 256 | CP-WOPT (Tensor Toolbox) | 12 | 575 | 7.517 | n/a | 3.199 |
| 256 | HaLRTC (XinyChen) | n/a | 575 | 7.517 | n/a | 0.000 |
| 256 | RTTC-TT (TTeMPS) | 12 | 575 | 7.517 | n/a | 3.061 |
| 256 | Tucker (TensorLy) | 12 | 575 | 7.517 | n/a | 0.001 |

### n = 5

| budget | method | rank/param | evals | off-grid RMSE | hidden RMSE | time(s) |
|---|---|---|---|---|---|---|
| 125 | Cheb-CP best | r=1, λ=0 | 150 | 483.1 | 824.2 | 0.868 |
| 125 | CP-WOPT (Tensor Toolbox) | 12 | 150 | 492.8 | 847.1 | 3.078 |
| 125 | HaLRTC (XinyChen) | n/a | 150 | 501.3 | 863.1 | 0.015 |
| 125 | Tucker (TensorLy) | 12 | 150 | 503.3 | 880.0 | 0.002 |
| 125 | RTTC-TT (TTeMPS) | 12 | 150 | 515.2 | 886.4 | 3.055 |
| 313 | CP-WOPT (Tensor Toolbox) | 12 | 388 | 67.66 | 118.2 | 3.180 |
| 313 | Cheb-CP best | r=30, λ=1 | 388 | 152.4 | 331.8 | 1.871 |
| 313 | HaLRTC (XinyChen) | n/a | 388 | 247.0 | 486.8 | 0.022 |
| 313 | Tucker (TensorLy) | 12 | 388 | 407.9 | 814.6 | 0.002 |
| 313 | RTTC-TT (TTeMPS) | 12 | 388 | 423.5 | 796.0 | 3.374 |
| 625 | Cheb-CP best | r=100, λ=0 | 1049 | 0.759 | n/a | 4.210 |
| 625 | CP-WOPT (Tensor Toolbox) | 12 | 1049 | 0.759 | n/a | 3.259 |
| 625 | HaLRTC (XinyChen) | n/a | 1049 | 0.759 | n/a | 0.000 |
| 625 | RTTC-TT (TTeMPS) | 12 | 1049 | 0.759 | n/a | 2.878 |
| 625 | Tucker (TensorLy) | 12 | 1049 | 0.759 | n/a | 0.002 |

### n = 6

| budget | method | rank/param | evals | off-grid RMSE | hidden RMSE | time(s) |
|---|---|---|---|---|---|---|
| 260 | CP-WOPT (Tensor Toolbox) | 12 | 320 | 475.3 | 728.9 | 3.125 |
| 260 | Tucker (TensorLy) | 12 | 320 | 479.5 | 783.8 | 0.002 |
| 260 | Cheb-CP best | r=5, λ=1 | 320 | 493.6 | 751.4 | 0.600 |
| 260 | HaLRTC (XinyChen) | n/a | 320 | 496.1 | 779.1 | 0.028 |
| 260 | RTTC-TT (TTeMPS) | 12 | 320 | 508.5 | 804.3 | 3.553 |
| 648 | Cheb-CP best | r=5, λ=1 | 978 | 74.22 | 77.30 | 0.677 |
| 648 | CP-WOPT (Tensor Toolbox) | 12 | 978 | 164.7 | 151.6 | 3.097 |
| 648 | HaLRTC (XinyChen) | n/a | 978 | 234.8 | 405.8 | 0.033 |
| 648 | Tucker (TensorLy) | 12 | 978 | 402.4 | 709.2 | 0.002 |
| 648 | RTTC-TT (TTeMPS) | 12 | 978 | 420.4 | 705.9 | 3.991 |
| 1276 | Cheb-CP best | r=50, λ=0 | 3098 | 0.072 | 0.097 | 5.601 |
| 1276 | HaLRTC (XinyChen) | n/a | 3098 | 0.114 | 0.614 | 0.006 |
| 1276 | CP-WOPT (Tensor Toolbox) | 12 | 3098 | 4.085 | 31.75 | 3.099 |
| 1276 | RTTC-TT (TTeMPS) | 12 | 3098 | 25.77 | 190.0 | 3.395 |
| 1276 | Tucker (TensorLy) | 12 | 3098 | 45.94 | 373.5 | 0.002 |

### n = 7

| budget | method | rank/param | evals | off-grid RMSE | hidden RMSE | time(s) |
|---|---|---|---|---|---|---|
| 481 | Tucker (TensorLy) | 12 | 516 | 507.9 | 794.1 | 0.003 |
| 481 | CP-WOPT (Tensor Toolbox) | 12 | 516 | 536.6 | 822.3 | 3.199 |
| 481 | Cheb-CP best | r=1, λ=1 | 516 | 536.8 | 819.3 | 0.638 |
| 481 | HaLRTC (XinyChen) | n/a | 516 | 544.3 | 833.0 | 0.037 |
| 481 | RTTC-TT (TTeMPS) | 12 | 516 | 564.4 | 828.0 | 3.464 |
| 1201 | CP-WOPT (Tensor Toolbox) | 12 | 1914 | 59.22 | 100.9 | 3.111 |
| 1201 | Cheb-CP best | r=5, λ=0.1 | 1914 | 74.71 | 127.3 | 0.720 |
| 1201 | HaLRTC (XinyChen) | n/a | 1914 | 312.5 | 541.5 | 0.050 |
| 1201 | Tucker (TensorLy) | 12 | 1914 | 408.8 | 768.6 | 0.003 |
| 1201 | RTTC-TT (TTeMPS) | 12 | 1914 | 435.1 | 760.6 | 4.060 |
| 2114 | Cheb-CP best | r=50, λ=0 | 4962 | 0.417 | 1.861 | 7.225 |
| 2114 | CP-WOPT (Tensor Toolbox) | 12 | 4962 | 18.69 | 50.19 | 3.138 |
| 2114 | HaLRTC (XinyChen) | n/a | 4962 | 47.62 | 151.2 | 0.048 |
| 2114 | Tucker (TensorLy) | 12 | 4962 | 229.8 | 744.7 | 0.003 |
| 2114 | RTTC-TT (TTeMPS) | 12 | 4962 | 235.0 | 673.9 | 4.158 |

### n = 8

| budget | method | rank/param | evals | off-grid RMSE | hidden RMSE | time(s) |
|---|---|---|---|---|---|---|
| 820 | CP-WOPT (Tensor Toolbox) | 12 | 940 | 265.0 | 530.3 | 3.254 |
| 820 | Cheb-CP best | r=50, λ=1 | 940 | 321.2 | 588.9 | 5.900 |
| 820 | Tucker (TensorLy) | 12 | 940 | 420.6 | 736.1 | 0.003 |
| 820 | HaLRTC (XinyChen) | n/a | 940 | 463.4 | 768.9 | 0.071 |
| 820 | RTTC-TT (TTeMPS) | 12 | 940 | 506.8 | 802.4 | 4.127 |
| 2048 | CP-WOPT (Tensor Toolbox) | 12 | 3007 | 56.56 | 83.76 | 3.072 |
| 2048 | Cheb-CP best | r=30, λ=0.1 | 3007 | 69.79 | 115.3 | 3.746 |
| 2048 | Tucker (TensorLy) | 12 | 3007 | 334.6 | 673.1 | 0.003 |
| 2048 | HaLRTC (XinyChen) | n/a | 3007 | 340.2 | 612.3 | 0.076 |
| 2048 | RTTC-TT (TTeMPS) | 12 | 3007 | 441.0 | 718.9 | 4.149 |
| 2640 | Cheb-CP best | r=30, λ=0 | 7804 | 1.148 | 1.711 | 3.943 |
| 2640 | CP-WOPT (Tensor Toolbox) | 12 | 7804 | 20.98 | 30.32 | 3.113 |
| 2640 | HaLRTC (XinyChen) | n/a | 7804 | 146.6 | 317.5 | 0.076 |
| 2640 | Tucker (TensorLy) | 12 | 7804 | 288.0 | 671.2 | 0.003 |
| 2640 | RTTC-TT (TTeMPS) | 12 | 7804 | 352.1 | 663.6 | 4.244 |

## AS+TT authors / paper-style baseline

| n | budget | method | status | evals | unique | rank | off-grid RMSE | off-grid max | time(s) |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 16 | AS+TT authors (converged) | ok | 184 | 16 | 1x4x4x2x1 | 410.9 | 1 016.1 | 38.39 |
| 2 | full | AS+TT authors full | ok | 184 | 16 | 1x4x4x2x1 | 410.9 | 1 016.1 | 37.48 |
| 3 | 81 | AS+TT authors (converged) | ok | 630 | 81 | 1x5x5x3x1 | 65.17 | 146.5 | 39.51 |
| 3 | full | AS+TT authors full | ok | 630 | 81 | 1x5x5x3x1 | 65.17 | 146.5 | 38.55 |
| 4 | 256 | AS+TT authors (converged) | ok | 1488 | 256 | 1x5x5x4x1 | 13.40 | 61.75 | 42.61 |
| 4 | full | AS+TT authors full | ok | 1488 | 256 | 1x5x5x4x1 | 13.40 | 61.75 | 41.81 |
| 5 | 625 | AS+TT authors (converged) | ok | 3050 | 625 | 1x5x5x5x1 | 11.13 | 57.45 | 49.01 |
| 5 | full | AS+TT authors full | ok | 3050 | 625 | 1x5x5x5x1 | 11.13 | 57.45 | 48.44 |
| 6 | 1276 | AS+TT authors (converged) | ok | 4392 | 1276 | 1x5x5x5x1 | 12.61 | 47.10 | 60.46 |
| 6 | full | AS+TT authors full | ok | 4392 | 1276 | 1x5x5x5x1 | 12.61 | 47.10 | 60.04 |
| 7 | full | AS+TT authors full | ok | 5978 | 2114 | 1x5x5x5x1 | 14.79 | 52.41 | 74.37 |
| 8 | 2640 | AS+TT authors (converged) | ok | 7808 | 2640 | 1x5x5x5x1 | 12.46 | 50.28 | 83.80 |
| 8 | full | AS+TT authors full | fail | 0 | 0 | n/a | n/a | n/a | 19.02 |

## Ошибки запуска

| n | baseline | budget | time(s) | short error |
|---|---|---|---|---|
| 7 | authors_budgeted | 2114 | 14.976102 | Octave TT-Toolbox run failed |
| 8 | authors_budgeted | 820 | 5.340780 | Octave TT-Toolbox run failed |
| 8 | authors_full | full | 19.024014 | Octave TT-Toolbox run failed |
| 9 | authors_sampling | 6561 | 52.181064 | Octave TT-Toolbox run failed |
| 10 | authors_sampling | 10000 | 29.135870 | Octave TT-Toolbox run failed |

## Выводы

1. **На максимальных доступных бюджетах Cheb-CP выглядит сильнее всех fixed-mask baselines.**  
   Особенно это видно для больших сеток:  
   - `n=6`: Cheb-CP best даёт `0.072 meV`, ближайший baseline HaLRTC — `0.114 meV`, CP-WOPT — `4.085 meV`;  
   - `n=7`: Cheb-CP best даёт `0.417 meV`, CP-WOPT — `18.69 meV`;  
   - `n=8`: Cheb-CP best даёт `1.148 meV`, CP-WOPT — `20.98 meV`.

2. **На малых и средних бюджетах CP-WOPT часто сильнее.**  
   CP-WOPT выигрывает при `n=3` на budgets `17` и `41`, при `n=4` на budget `128`, при `n=5` на budget `313`, при `n=6` на budget `260`, при `n=7` на budget `1201`, а также при `n=8` на budgets `820` и `2048`. Это значит, что Cheb-CP раскрывается лучше при достаточно плотном префиксе author trace.

3. **Cheb-CP очень чувствителен к rank и λ.**  
   В исходном отчёте часть конфигураций Cheb-CP давала огромные ошибки. Поэтому для статьи нельзя показывать только “лучший из 15” без честного protocol selection. Лучше заранее зафиксировать схему выбора rank/λ по validation или показывать отдельный hyperparameter sweep.

4. **AS+TT authors — важный paper-style baseline, но это другой класс сравнения.**  
   Он использует adaptive TT-cross и часто требует больше фактических PES evaluations. Например, при `n=6` AS+TT authors full даёт `12.61 meV` при `4392` total evals, а Cheb-CP best на fixed mask даёт `0.072 meV` при `3098` evals. Это сильный результат, но его нужно описывать аккуратно: Cheb-CP использует точки из author-selected trace.

5. **Результаты для `n=2–3` малоинформативны.**  
   Там сетка слишком маленькая, многие методы почти совпадают на полном/почти полном grid. Основную аргументацию лучше строить по `n=5–8`.

6. **Запуски `n=8 full`, `n=9`, `n=10` требуют повторной проверки Octave/TT-Toolbox bridge.**  
   В отчёте есть падения `authors_full`/`authors_sampling`; эти строки нельзя использовать как численные результаты.
