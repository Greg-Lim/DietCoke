import lavis
import torch
from lavis.models import load_model_and_preprocess
import fastapi
from fastapi import UploadFile, File, Form
from PIL import Image
import io


app = fastapi.FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, txt_processors = load_model_and_preprocess(name="img2prompt_vqa", model_type="base", is_eval=True, device=device)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate_caption_qa")
async def generate_caption_qa(file: UploadFile = File(...), question: str = Form(...)):
    print("file:", file)
    print("question:", question)
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    lavis_image = vis_processors["eval"](image).unsqueeze(0).to(device)
    lavis_question = txt_processors["eval"](question)
    lavis_samples = {"image": lavis_image, "text_input": [lavis_question]}
    lavis_samples = model.forward_itm(samples=lavis_samples)
    lavis_samples = model.forward_cap(samples=lavis_samples, num_captions=50, num_patches=20)
    lavis_samples = model.forward_qa_generation(lavis_samples)

    return lavis_samples
