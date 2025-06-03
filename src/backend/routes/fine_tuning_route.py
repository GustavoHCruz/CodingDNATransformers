from fastapi import APIRouter
from services.decorators import standard_response

router = APIRouter(prefix="/fine-tuning", tags=["Fine Tuning"])

@router.get("/load")
@standard_response()
def load_model():
  return

@router.post("/train")
@standard_response()
def train_model():
  return

@router.post("/evaluate")
@standard_response()
def evaluate_model():
  return

@router.post("/save")
@standard_response()
def save_model():
  return

