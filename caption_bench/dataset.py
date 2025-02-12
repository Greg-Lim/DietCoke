import json
import PIL.Image

path = "dataset/ok_vqa_data"
image_root_path = "dataset/ok_vqa_data/train2014"
vqa_annotations_path = "dataset/ok_vqa_data/mscoco_train2014_annotations.json"
vqa_question_path = "dataset/ok_vqa_data/OpenEnded_mscoco_train2014_questions.json"

with open(vqa_annotations_path, 'r') as file:
    vqa_annotations = json.load(file)

with open(vqa_question_path, 'r') as file:
    vqa_questions = json.load(file) 

def get_data(count: int): 
    data = []

    for idx in range(count):
        image_path = f"{image_root_path}/COCO_train2014_{vqa_questions['questions'][idx]['image_id']:012}.jpg"
        image = PIL.Image.open(image_path)
        data.append({
            "image_path": image_path,
            "image": image,
            "question": vqa_questions["questions"][idx]["question"],
            "answer": vqa_annotations["annotations"][idx]["answers"],
            "question_id": vqa_questions["questions"][idx]["question_id"]
        })
    
    return data

if __name__ == "__main__":
    t = get_data(2)
    print(t[0])
