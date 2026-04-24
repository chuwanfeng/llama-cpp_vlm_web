"""模拟前端传 JPEG 图片的完整链路"""
import sys, io, os, base64, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, r'D:\vps\python\llama-cpp_vlm_web')

from gpu_backend import load_model, infer, unload_model
from PIL import Image
from config import MODELS_DIR

model_path = os.path.join(MODELS_DIR, 'qwen3.5-4B-Q4', 'Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf')

# 创建一个 660x946 的测试 JPEG（模拟红毯照片尺寸）
img = Image.new('RGB', (660, 946), color=(200, 50, 50))
buf = io.BytesIO()
img.save(buf, format='JPEG', quality=90)
jpeg_bytes = buf.getvalue()
print('Original JPEG size: ' + str(len(jpeg_bytes)) + ' bytes')

# 模拟前端: FileReader.readAsDataURL -> "data:image/jpeg;base64,XXXXX"
frontend_data_uri = 'data:image/jpeg;base64,' + base64.b64encode(jpeg_bytes).decode()
print('Frontend data URI length: ' + str(len(frontend_data_uri)))

# 模拟后端 _img_to_bytes: 解码 base64
decoded = base64.b64decode(frontend_data_uri.split(',', 1)[1])
print('Decoded bytes match original: ' + str(decoded == jpeg_bytes))

# 模拟后端 infer: 重新编码为 PNG data URI
re_b64 = base64.b64encode(decoded).decode()
re_uri = 'data:image/png;base64,' + re_b64
print('Re-encoded data URI length: ' + str(len(re_uri)))

# 用这个 data URI 调用 infer
print('Loading model...')
load_model(model_path=model_path, chat_handler='auto')

print('Testing with JPEG-as-PNG data URI (matching app.js flow)...')
result = infer(
    prompt='What is the dominant color of this image? Answer in one word.',
    images=[frontend_data_uri],  # 直接传前端的 data URI
    stream=False,
    max_tokens=100,
)
print('Result (direct frontend data URI): ' + result.strip())

print('')
print('Testing with decoded+reencoded (matching gpu_backend infer logic)...')
result2 = infer(
    prompt='What is the dominant color of this image? Answer in one word.',
    images=[re_uri],  # 传经过 _img_to_bytes + 重新编码的 URI
    stream=False,
    max_tokens=100,
)
print('Result (re-encoded): ' + result2.strip())

unload_model()
print('Done')
