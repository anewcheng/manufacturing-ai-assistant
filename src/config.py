from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")  # or "gpt-3.5-turbo"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # HuggingFace token 或 OpenAI key
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"

# 本地模型 (ollama 或 vLLM)
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "models/codellama_finetuned")
LOCAL_API_URL = os.getenv("LOCAL_API_URL", "http://localhost:8000")

# 單獨指定本地模型名，新增一個：
# LOCAL_MODEL_NAME = os.getenv("LLM_MODEL", LLM_MODEL)
LOCAL_MODEL_NAME = LLM_MODEL

# 應用設定
APP_TITLE = "Manufacturing AI Code & Report Assistant"
MAX_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.95

# 資料設定
SUPPORTED_FORMATS = ["csv", "xlsx", "json"]
MAX_FILE_SIZE_MB = 100

# 快取設定 (Streamlit caching)
CACHE_TTL_SECONDS = 3600

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
# 將字串路徑轉換為 Path 物件
LOG_DIR = Path("logs") 
LOG_DIR.mkdir(exist_ok=True) # 確保 logs 目錄存在
LOG_FILE = LOG_DIR / "app.log" # 更清晰的跨平台路徑組合

# 本地模型 (ollama 或 vLLM)
# 假設 LOCAL_MODEL_PATH 也是一個路徑
LOCAL_MODEL_PATH = Path(os.getenv("LOCAL_MODEL_PATH", "models/codellama_finetuned"))
