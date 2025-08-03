from dotenv import load_dotenv
from src.llms.exin_classifier.gpt import create_model, train_model

load_dotenv()

create_model("gpt2", "EXIN-001", "uuid")

train_model("EXIN-001", "uuid", 30000, 1, 2, 8, 2e-5, 0.01, 0.2, 1234)
