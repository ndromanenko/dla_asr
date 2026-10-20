# ASR: Conformer (CTC) на LibriSpeech

## Ссылки

- [Отчёт 1](https://www.comet.com/niki-romanenko/pytorch-template-asr-example/reports/mrkFp7k64fiwtzhG3MEkpAaRo)
- [Отчёт 2](https://www.comet.com/niki-romanenko/asr-train/reports/BmIao0WdIr6R6dVrfqhQhvb22)
- [Аугментации](https://www.comet.com/niki-romanenko/asr-train/e3a6cffa565a4d2a8cb953c792dcab2c?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&viewId=new&xAxis=step)
- [Тестовые данные](https://drive.google.com/file/d/1WoPJZKyRnyRlQcyDDXIm1qsJh4ClRWIX/view?usp=sharing)
- [Модель](https://huggingface.co/romanenko02/conformer-asr) — скачивается скриптом `uv run python download_model.py` в `ckpt/checkpoint_60epoch.pth`

## Установка

```bash
git clone https://github.com/ndromanenko/dla_asr.git
cd dla_asr
pip install uv
uv python install 3.12
uv sync --python 3.12 --locked
```

## Обучение

1. Залогиниться в CometML:
   ```bash
   uv run comet login
   ```
2. Запустить обучение:
   ```bash
   uv run python train.py -cn conformer_s writer.run_name=train_init_60_s
   ```

## Инференс

Инференс показан в `demo.ipynb` — загрузите ноутбук в Google Colab и запускайте ячейки по порядку. Ноутбук сам клонирует репозиторий, ставит зависимости, скачивает чекпоинт (`download_model.py`, в `ckpt/`) и тестовые данные.

Есть 3 варианта инференса — выберите один, остальные закомментируйте:

| Вариант | Метрики | Сохраняемые предсказания |
|---|---|---|
| only argmax | Argmax CER/WER | argmax |
| only beam search | Beam search CER/WER | beam search |
| beam search + argmax | все 4 | beam search |

После инференса ячейка с `calc_metrics.py` считает CER/WER по сохранённым предсказаниям (`data/saved/predictions/test`) и транскрипциям.

Формат данных для инференса:
```
test_audio/
├── audio/            # *.flac / *.wav / *.mp3 / *.m4a
└── transcriptions/   # <имя_аудио>.txt (опционально)
```
Если транскрипций нет (папка пустая), инференс отработает и сохранит предсказания, но все метрики будут равны 1.
