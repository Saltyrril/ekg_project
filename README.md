# Классификация ЭКГ на PTB-XL
Исследовательский прототип (не медицинское изделие): multi-label классификация по 5 суперклассам (NORM, MI, STTC, CD, HYP).

## Установка
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python scripts/check_env.py   # интерпретатор в .venv, версии из requirements.txt, torch + CUDA
python scripts/check_cuda.py  # подробная проверка GPU
```
Данные PTB-XL положите в `data/` (версия 100 Гц).

## Итоговый результат (test, strat_fold 10, однократная оценка)
Train — фолды 1–8, val — фолд 9 (выбор модели, early stopping, температура T и пороги), test — фолд 10 (оценён один раз,
ничего не подбиралось). 2158 записей, 1877 пациентов. В скобках — 95% ДИ: перцентильный bootstrap по пациентам
(2000 реплик), разница — парный bootstrap на тех же репликах.

| Метрика | 1D-CNN + аугментации | LightGBM по признакам | разница |
|---|---|---|---|
| macro AUROC | 0.922 [0.914, 0.929] | 0.880 [0.871, 0.889] | +0.042 [0.035, 0.049] |
| macro PR-AUC | 0.811 [0.794, 0.827] | 0.739 [0.719, 0.757] | +0.073 [0.058, 0.087] |
| macro F1 (пороги с val) | 0.743 [0.727, 0.757] | 0.666 [0.649, 0.681] | +0.076 [0.063, 0.091] |
| AUROC HYP | 0.899 [0.879, 0.917] | 0.886 [0.862, 0.907] | +0.013 [−0.001, 0.028] |
| macro Brier (после T) | 0.087 [0.083, 0.092] | 0.107 [0.103, 0.112] | −0.020 [−0.023, −0.017] |

Выводы:
- CNN лучше LightGBM по AUROC, PR-AUC и F1: 95% ДИ разницы не содержат нуля.
- По HYP (самый редкий класс, 12%) преимущества нет: ДИ разницы AUROC включает ноль.
- По калибровке вывода нет: Brier лучше у CNN, а ECE надёжно не оценён (его bootstrap-ДИ смещены вверх).

## Ограничения
- Исследовательский прототип, не медицинское изделие: не предназначен для диагностики и клинических решений.
- Val-метрики (фолд 9) оптимистичны: по ним выбирались модель, эпоха, T и пороги. Несмещённая оценка — только test выше.
- ECE не оценён надёжно: bootstrap-ДИ бинированного ECE смещены вверх, вывода о калибровке нет.
- По HYP преимущества CNN над LightGBM нет.
- Один seed (42) на модель: разброс между запусками не оценён, разницы ≤0.005 AUROC между вариантами на val — в пределах шума.
- T и пороги бейзлайна подобраны на val той же процедурой, что у CNN; это решение принято после согласования плана
  (см. `docs/decisions.md`).

Кэш данных сверен с CSV PTB-XL (`scripts/verify_cache.py`, `results/final/cache_check.json`). Все метрики и примечания — в `results/final/table.md`
и `results/final/metrics.json`. Воспроизвести: `python -m src.final_eval` (повторный запуск в ту же папку запрещён).

## Демо
Статическая страница с пятью заранее посчитанными примерами из val (фолд 9), без весов и сырых сигналов:
[saltyrril.github.io/ekg_project/demo](https://saltyrril.github.io/ekg_project/demo/) (папка `docs/demo/`,
собирается `python scripts/export_static_demo.py --config configs/static_demo.yaml`, нужен собранный `app/model`).

### Gradio, Hugging Face Spaces
Папка `app/` служит корнем Space: `app.py` (интерфейс), `inference.py` (предобработка, модель, Integrated Gradients),
`README.md` (метаданные Space), `requirements.txt` (CPU-torch, gradio). Веса и примеры в git не хранятся, их собирает скрипт.
Он же проверяет, что приложение воспроизводит `val_probs.npy` запуска (только val, test не читается):
```powershell
pip install gradio==6.31.0                                     # в .venv обучения ставьте только gradio (см. ниже)
python scripts/export_app_bundle.py --config configs/app.yaml  # app/model, app/examples, app/ecg_models.py
python app/app.py                                              # http://127.0.0.1:7860
```
`app/requirements.txt` рассчитан на чистое окружение (Space или отдельный venv). Если поставить его в `.venv` обучения,
CUDA-сборка torch заменится на CPU-сборку.

Деплой (один раз `hf auth login` с токеном write; `hf` ставится вместе с gradio):
```powershell
hf repos create <user>/<space> --type space --sdk gradio
hf upload <user>/<space> app . --repo-type space --exclude "__pycache__/*" --exclude ".gitkeep"
```
