# Обучение моделей `lu_sol` и `lu_astra` на Kaggle

Здесь лежит всё, чтобы обучить максимальные Classic-политики проекта
(`lu_sol` и более сильную `lu_astra`) на внешних мощностях, а не на ноутбуке.

```
kaggle/
├── lu_sol-training.ipynb    # готовый ноутбук: клонирует репо и запускает обучение lu_sol
├── train_lu_sol.py          # возобновляемый драйвер обучения lu_sol
├── lu_astra-training.ipynb  # готовый ноутбук для lu_astra (12 ч, GPU)
├── train_lu_astra.py        # возобновляемый драйвер обучения lu_astra
└── README.md                # этот файл
```

> `lu_astra` — тот же движок и тот же формат JSON, что у `lu_sol`, но с более
> широкой сетью (1024/512/512), 12 эпохами PPO и отбором лучшего чекпоинта по
> победе над `lu_sol`. См. раздел «Модель `lu_astra`» ниже.

## Почему Kaggle и как используется GPU

У проекта два бэкенда обучения:

- **`numpy`** — эталонный: чистый Python + NumPy, тот же код, что играет в
  приложении. Узкое место — поштучный проигрыш игр, поэтому помогают только
  ядра CPU (`--workers 4`).
- **`torch`** — векторный: симулятор и сеть переписаны на PyTorch, тысячи игр
  играются одним батчем на GPU (или CPU), PPO считается автоградом. Веса
  экспортируются в тот же JSON, который грузит приложение, — на деплой это
  никак не влияет.

На CPU векторный бэкенд уже примерно **в 40 раз быстрее** эталонного
(≈3 мс/игру против ≈127 мс/игру для сети 512/256/256 и `v2`), а на GPU T4/P100
добавляется ещё больше. Поэтому на Kaggle включайте **Accelerator: GPU T4 x2**
и запускайте с `--backend torch --device cuda`. Если GPU недоступен, тот же
код идёт на CPU (`--backend numpy --workers 4` или `--backend torch --device cpu`).

Каждая сессия Kaggle — примерно 12 часов. Полный прогон не влезает в одну
сессию, поэтому драйвер учит **чанками** и продолжает в новой сессии с того
же шага.

## Быстрый путь (ноутбук)

1. Откройте [kaggle.com](https://www.kaggle.com) → **Create → New Notebook**.
2. **File → Import Notebook** → загрузите `lu_sol-training.ipynb`.
3. В панели **Settings** справа:
   - **Internet: On** (иначе `git clone` не пройдёт);
   - **Accelerator: GPU T4 x2** (подойдёт и P100).
4. **Save Version → Save & Run All (Commit)**.
5. Когда сессия закончится, загляните в **Output**: заберите
   `lu_sol.json` (лучшая модель) и `lu_sol-metrics.json`.
6. Чтобы продолжить — запустите ноутбук снова: он возобновит обучение.

## Ручной путь (терминал, любой Linux)

```bash
git clone https://github.com/leonka123/8FM_AI
cd 8FM_AI

# один прогон: чанк за чанком, пока не кончится бюджет времени
python kaggle/train_lu_sol.py \
    --work-dir /kaggle/working \
    --backend torch \
    --device cuda \
    --total-updates 6000 \
    --chunk-updates 200 \
    --games-per-update 512 \
    --warm-start-games 6000 \
    --time-budget-seconds 30000
```

Без GPU используйте `--backend numpy --workers 4 --games-per-update 48`
(или `--backend torch --device cpu`).

Если сессия прервалась — просто запустите команду ещё раз. Драйвер сам
прочитает `driver_state.json` и продолжит с того же глобального шага.

### Полезные флаги

| Флаг | Значение | По умолчанию |
|---|---|---|
| `--work-dir` | где лежат чекпоинты и состояние | `/kaggle/working` |
| `--backend` | `auto`, `numpy` или `torch` (`auto` = torch, если установлен) | `auto` |
| `--device` | устройство torch: `auto`, `cpu`, `cuda`, `cuda:0`, `mps` | `auto` |
| `--total-updates` | сколько всего апдейтов обучать | `6000` |
| `--chunk-updates` | размер одного чанка между сохранениями | `200` |
| `--games-per-update` | игр на апдейт | `48` numpy / `512` torch |
| `--workers` | параллельных процессов сбора игр (только numpy) | `4` |
| `--warm-start-games` | игр для behaviour cloning при старте с нуля | `6000` |
| `--time-budget-seconds` | остановиться до лимита сессии (`0` = без лимита) | `30000` |
| `--min-chunk-seconds` | не начинать чанк, если осталось меньше | `600` |
| `--reset` | забыть прошлое состояние и начать заново | выкл. |

## Что получается на выходе

В `--work-dir` появятся:

- `lu_sol.json` — **лучшая** модель по композитному скору
  (её и переносите в проект);
- `lu_sol-latest.json` — последняя сеть (нужна только для продолжения
  обучения);
- `lu_sol-metrics.json` — метрики по всем чанкам и сессиям
  (можно рисовать графики);
- `driver_state.json` — служебный файл с глобальным шагом и лучшим скором;
- `checkpoints/` — архив снимков каждые `--checkpoint-every` обновлений.

## Перенос модели на ноутбук

1. Скачайте `lu_sol.json` из **Output** (или скопируйте с `--work-dir`).
2. Положите в локальный проект, например в `models/lu_sol.json`.
3. Пользуйтесь как обычно:

```powershell
python -X utf8 -m eightfm eval-rl --model models/lu_sol.json --games 100 --seed-base 90000
python -X utf8 -m eightfm play-game --model models/lu_sol.json
```

> **Совместимость.** `lu_sol` использует расклад признаков `v2`
> (158 / 32), старая `lu_luna` — `v1` (115 / 25). Они не взаимозаменяемы как
> чекпоинты: продолжайте `lu_sol` только из `lu_sol`. При этом обе модели
> спокойно сосуществуют — сеть помнит свою версию признаков и грузится как есть,
> а в одной игре они могут играть друг против друга.

## Сколько это займёт

Замеры на локальной CPU-машине с сетью `lu_sol` (512/256/256) и раскладом `v2`:

| Операция | Время |
|---|---|
| Сбор игр, эталонный NumPy-бэкенд | ~127 мс/игру |
| Сбор игр, torch-бэкенд на CPU (батч 512) | ~3 мс/игру (**≈40×**) |
| PPO-апдейт (батч 1024, 8 эпох) | доли секунды |
| Полная оценка (60 игр × 4 противника × 5 форматов + 40 личных встреч) | ~2 мин (последовательно) |

На GPU T4/P100 векторный сбор и PPO идут ещё быстрее; точная цифра зависит
от карты, поэтому ориентируйтесь на счётчик `games` в логе драйвера.
Оценка по-прежнему считается на NumPy и стоит около 2 минут, но запускается
раз в 50 апдейтов (`--eval-every`). С батчем 512 игр один апдейт на GPU —
это секунды, так что 6000 апдейтов реально пройти за одну–две сессии с
учётом `--time-budget-seconds`. Хотите быстрее — уменьшите
`--total-updates`; хотите сильнее — увеличивайте.

## Восстановление после обрыва

Драйвер специально пишет `latest`-чекпоинт каждые несколько апдейтов. Даже
если процесс убьют жёстко, следующий запуск:
1. прочитает `driver_state.json` (или, если его нет, начнёт с шага 0);
2. загрузит `lu_sol-latest.json`;
3. продолжит с сохранённым learning rate и энтропией.

Расписание `learning rate`/`энтропии` вычисляется от **глобального** шага,
поэтому при возобновлении оно не «перезапускается», а плавно продолжается.

## Модель `lu_astra`

`lu_astra` — самая сильная развёртываемая политика проекта. Она использует
тот же проверенный движок и тот же JSON-формат `RLNetwork`, что и `lu_sol`,
поэтому чекпоинт сразу работает в приложении и турнире. Отличия:

| Параметр | `lu_sol` | `lu_astra` |
|---|---|---|
| сеть (hidden1/hidden2/action) | 512/256/256 | **1024/512/512** |
| всего обновлений | 6000 | **8000** |
| PPO-эпох на батч | 8 | **12** |
| warm start | 6000 игр / 16 эпох | **12000 игр / 24 эпохи** |
| лига | 65% / пул 12 | **75% / пул 16** |
| отбор лучшего | против `classic-rl` | **против `lu_sol`** |
| learning rate | 3e-4 → 2e-5 | **4e-4 → 1e-5** |
| entropy | 0.035 → 0.003 | **0.04 → 0.002** |

### Быстрый путь (ноутбук)

1. Kaggle → **Create → New Notebook** → **File → Import Notebook** →
   `kaggle/lu_astra-training.ipynb`.
2. **Settings**: **Internet: On**, **Accelerator: GPU T4 x2** (или P100).
3. **Save Version → Save & Run All (Commit)**.
4. Заберите из **Output** `lu_astra.json`, `lu_astra-latest.json`,
   `lu_astra-metrics.json` и `tournament_astra_report.json`.

### Ручной путь

```bash
git clone https://github.com/leonka123/8FM_AI
cd 8FM_AI
python kaggle/train_lu_astra.py \
    --work-dir /kaggle/working \
    --total-updates 8000 \
    --chunk-updates 200 \
    --time-budget-seconds 39600
```

Драйвер сам подхватит `lu_astra-latest.json` и `driver_state.json` при
повторном запуске. `--time-budget-seconds 39600` (11 ч) оставляет запас, чтобы
сессия успела сохранить чекпоинты до лимита Kaggle (~12 ч).

Полезные флаги: `--hidden1/--hidden2/--action-hidden` (ширина сети),
`--ppo-epochs`, `--league-fraction`, `--snapshot-pool-size`,
`--selection-opponent` (по умолчанию `models/lu_sol.json`, `none` — отключить),
`--eval-every`, `--eval-games`, `--device`, `--seed`, `--reset`.

Локальный smoke-тест (несколько секунд на CPU):

```bash
python -X utf8 -m eightfm train-lu-astra --updates 2 --games-per-update 8 \
    --players 3 --device cpu --hidden1 64 --hidden2 32 --action-hidden 32 \
    --warm-start-games 4 --bc-epochs 1 --eval-every 0 --selection-opponent none \
    --checkpoint-every 0 --plot none
```

### Турнир после обучения

```bash
python -X utf8 -m eightfm tournament \
    --participants luna,sol,astra,supervised,greedy,rl_classic \
    --astra models/lu_astra.json \
    --players 3,4,5,6,7 --games 100 \
    --output tournament_astra_report.json
```
