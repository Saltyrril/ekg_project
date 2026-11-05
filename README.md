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

## Демо (Gradio, Hugging Face Spaces)
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
