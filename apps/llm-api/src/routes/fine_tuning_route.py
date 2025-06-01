from fastapi import APIRouter

router = APIRouter(prefix="/fine-tuning", tags=["Fine Tuning"])

@router.get("/load")
def load_model():
  return

@router.post("/train")
def train_model():
  return

@router.post("/evaluate")
def evaluate_model():
  return

@router.post("/save")
def save_model():
  return

