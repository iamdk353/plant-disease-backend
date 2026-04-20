import os
 
MODEL_PATH     = os.getenv("MODEL_PATH", "./app/model/model.keras")
LABEL_MAP_PATH = os.getenv("LABEL_MAP_PATH", "./app/model/class.json")

# Leaf/Plant gate classifier (binary sigmoid model)
# Convention: prob >= threshold => plant/leaf
CLASSIFIER_PATH      = os.getenv("CLASSIFIER_PATH", "./app/model/classifier.keras")
CLASSIFIER_THRESHOLD = float(os.getenv("CLASSIFIER_THRESHOLD", "0.5"))
# If "1", app fails startup if classifier file is missing
CLASSIFIER_REQUIRED  = os.getenv("CLASSIFIER_REQUIRED", "0").strip() in ("1", "true", "True")

OCI_BUCKET     = os.getenv("OCI_BUCKET", "agri-images")
IMG_SIZE       = (300, 300)
TOP_K          = int(os.getenv("TOP_K", "5"))

# AI Agent configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
