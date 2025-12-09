import os
from fast_bitrix24 import BitrixAsync

webhook = os.getenv("BITRIX24_WEBHOOK_URL")
if not webhook:
    raise RuntimeError("BITRIX24_WEBHOOK_URL не установлен в переменных окружения")

b = BitrixAsync(webhook)