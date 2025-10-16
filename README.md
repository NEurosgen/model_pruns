# model_pruns

Набор утилит для ускорения и сжатия PyTorch-моделей через прунинг и квантизацию. Репозиторий
содержит готовые функции для пост-тренировочного тюнинга, прогрессивного прореживания каналов,
а также инструменты измерения латентности и отчётов по разреженности.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision
pip install -r requirements.txt  # если требуется
```

> **Совет.** Большинство функций использует `torch_pruning` и `omegaconf`. Убедитесь, что они
> установлены: `pip install torch-pruning omegaconf`.

## Быстрый старт

Ниже показан минимальный пример того, как измерить латентность модели, прогрессивно проредить
каналы и затем выполнить пост-тренировочную статическую квантизацию INT8.

```python
import torch
from torchvision.models import resnet18

from src.quantization.quant import measure_latency, ptq_static_int8, apply_smooth_quant
from src.pruning.channel_pruning import progressive_channel_pruning

model = resnet18(weights=None).eval()
example_inputs = torch.randn(1, 3, 224, 224)

print(f"Baseline latency: {measure_latency(model):.4f}s")

# Прогрессивно увеличиваем разреженность каналов (10% → 20%)
progressive_channel_pruning(
    model,
    example_inputs,
    schedule=[0.10, 0.20],
    global_pruning=False,
    min_out_channels=16,
    importance="taylor",
)

print(f"After pruning: {measure_latency(model):.4f}s")

# Калибруем SmoothQuant и затем запускаем статическую квантизацию
calib_data = [torch.randn(1, 3, 224, 224) for _ in range(32)]
apply_smooth_quant(model, calib_data, alpha=0.5, max_batches=32)
quantized = ptq_static_int8(model, calib_data, num_calib_batches=32)

print(f"INT8 latency: {measure_latency(quantized):.4f}s")
```

## Прунинг тензоров

Модуль `src/pruning/tensor_prunning.py` содержит несколько стратегий разреживания:

```python
from src.pruning import tensor_prunning as tp

# Глобальный L1-прунинг 30% параметров в Linear и Conv2d-слоях
tp.global_unstructured_prune(model, amount=0.3)

# Независимый прунинг каждого слоя
tp.layer_unstructured_prune(model, amount=0.2)

# Структурный прунинг фильтров в свёрточных слоях
tp.structured_filter_prune(model, amount=0.3, n=2, dim=0)

# Случайный прунинг (полезен для бенчмарков)
tp.random_unstructured_prune(model, amount=0.1)

# Жёсткое обнуление весов по порогу
threshold = 1e-3
tp.threshold_prune(model, threshold, prune_bias=True)

# Отчёт по разреженности и фиксация масок
stats = tp.report_sparsity(model)
tp.make_pruning_permanent(model)
```

## Квантизация

Файл `src/quantization/quant.py` предоставляет несколько вариантов квантизации:

- `dynamic_quantize_linear` — динамическая INT8 квантизация линейных слоёв (без калибровки).
- `ptq_static_int8` — статическая пост-тренировочная квантизация с калибровкой.
- `apply_smooth_quant` — SmoothQuant-подобное масштабирование для стабилизации INT8.
- `quantize_fp16` / `quantize_bf16` — лёгкая весовая квантизация в FP16/BF16.

Пример статической квантизации с кастомным набором слоёв:

```python
from torch.utils.data import DataLoader

calib_loader = DataLoader(calib_dataset, batch_size=32)
quantized = ptq_static_int8(
    model,
    calib_loader,
    backend="x86",
    per_channel_weights=True,
    num_calib_batches=100,
    device="cpu",
)
```

## Измерение производительности и отчёты

```python
from src.quantization.quant import measure_latency
from src.pruning.tensor_prunning import report_sparsity

latency = measure_latency(model, input_size=(1, 3, 224, 224))
print(f"Latency: {latency:.6f}s")

stats = report_sparsity(model)
for name, sparsity in stats.items():
    print(f"{name}: {sparsity:.2%}")
```

## Загрузка моделей из чекпоинтов

Утилита `src/checkpoints_utils/checkpoint_load.py` восстанавливает модели из чекпоинтов
PyTorch Lightning. Передайте путь к `.ckpt`, конфигурацию `OmegaConf` (или позвольте функции
получить её из чекпоинта), и затем используйте полученную модель для прунинга/квантизации.

```python
from src.checkpoints_utils.checkpoint_load import load_model_from_checkpoint

model = load_model_from_checkpoint(
    "path/to/checkpoint.ckpt",
    cfg=my_cfg,
    num_classes=1000,
)
```

## Рекомендации по пайплайну

1. **Снимите бэйзлайн.** Измерьте латентность и точность до оптимизации.
2. **Прогрессивный прунинг.** Используйте `progressive_channel_pruning` или комбинации
   функций из `tensor_prunning` для аккуратного удаления нейронов/фильтров.
3. **Дообучение.** После каждого шага прунинга рекомендуется сделать короткий fine-tune.
4. **Квантизация.** Примените `apply_smooth_quant`, затем `ptq_static_int8` для лучшего баланса
   между размером и точностью. Для быстрой проверки можно начать с `dynamic_quantize_linear`.
5. **Фиксация и экспорт.** Закрепите маски `make_pruning_permanent`, сохраните квантизованную
   модель и переснимите метрики.

Все функции можно использовать из Python API или встроить в собственные скрипты обучения.
