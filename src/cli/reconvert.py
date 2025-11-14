# save as reconvert_qnnpack.py and run on Raspberry
import torch, cv2
import numpy as np
import torchvision
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

# 0) ВАЖНО: бэкенд ARM
torch.backends.quantized.engine = "qnnpack"

# 1) Загружаем FP32 модель и веса (твои исходные float-веса!)
model = torchvision.models.mobilenet_v3_large(weights=None).eval()
state = torch.load("fp32_mnv3_large.pth", map_location="cpu")  # <-- твой FP32 чекпойнт
model.load_state_dict(state, strict=False)
model = torch.jit.load("/home/temp/MyDir/Projects/aspdfpwjfpwejfwpefwef/saved_model/mnv3_dynamic_int8_scripted.pt")
# 2) Конфиг квантовки
qconfig = get_default_qconfig_mapping("qnnpack")

example = torch.randn(1,3,224,224)
prepared = prepare_fx(model, {"": qconfig}, example_inputs=example).eval()

# 3) Препроцесс (как при обучении/калибровке!)
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_bgr(frame_bgr, size=224):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    x = (x - mean) / std
    x = np.transpose(x, (2,0,1))[None, ...]  # NCHW
    return torch.from_numpy(x)  # float32

# 4) Калибровка на реальных кадрах (без обучения!)
cap = cv2.VideoCapture("/home/temp/MyDir/Projects/ml_things/Results/results/processed_sample_EfficientNetB0_TL.avi" )
n = 0
with torch.inference_mode():
    while cap.isOpened() and n < 200:   # 100–300 кадров обычно достаточно
        ret, frame = cap.read()
        if not ret: break
        x = preprocess_bgr(frame)
        prepared(x)
        n += 1
cap.release()
print("Calibrated on", n, "frames")

# 5) Конвертация в INT8 под QNNPACK и сохранение TorchScript
qmodel = convert_fx(prepared).eval()
torch.jit.script(qmodel).save("mnv3_static_int8_qnnpack.pt")
print("Saved mnv3_static_int8_qnnpack.pt")
