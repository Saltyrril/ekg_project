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
