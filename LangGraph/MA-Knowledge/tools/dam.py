import torch
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor, AutoModel
import cv2
import requests
from io import BytesIO

device = torch.device("cpu")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

model = AutoModel.from_pretrained(
    'nvidia/DAM-3B-Self-Contained',
    trust_remote_code=True,
    torch_dtype='torch.float16'
).to("cuda")
dam = model.init_dam(conv_mode='v1', prompt_mode='full+focal_crop')


def apply_sam(image, input_points=None, input_boxes=None, input_labels=None):
  inputs = sam_processor(image, input_points=input_points, input_boxes=input_boxes,
                         input_labels=input_labels, return_tensors="pt").to(device)

  with torch.no_grad():
    outputs = sam_model(**inputs)

  masks = sam_processor.image_processor.post_process_masks(
      outputs.pred_masks.cpu(),
      inputs["original_sizes"].cpu(),
      inputs["reshaped_input_sizes"].cpu()
  )[0][0]
  scores = outputs.iou_scores[0, 0]

  mask_selection_index = scores.argmax()
  mask_np = masks[mask_selection_index].numpy()
  return mask_np


def add_contour(img, mask, input_points=None, input_boxes=None):
  img = img.copy()
  mask = mask.astype(np.uint8) * 255
  contours, _ = cv2.findContours(
      mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(img, contours, -1, (1.0, 1.0, 1.0), thickness=6)

  if input_points is not None:
    for points in input_points:
      for x, y in points:
        cv2.circle(img, (int(x), int(y)), radius=10,
                   color=(1.0, 0.0, 0.0), thickness=-1)
        cv2.circle(img, (int(x), int(y)), radius=10,
                   color=(1.0, 1.0, 1.0), thickness=2)

  if input_boxes is not None:
    for box_batch in input_boxes:
      for box in box_batch:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2),
                      color=(1.0, 1.0, 1.0), thickness=4)
        cv2.rectangle(img, (x1, y1), (x2, y2),
                      color=(1.0, 0.0, 0.0), thickness=2)

  return img

def print_streaming(text):
  print(text, end="", flush=True)


def run_full_image_vqa(
    image_url: str,
    question: str,
) -> str:
  """
  Tải ảnh từ `image_url`, tạo full‐mask, visualize nếu cần,
  rồi gọi DAM để trả về 5 đáp án hàng đầu cho `question`.

  Nếu save_vis_path được truyền, lưu ảnh contour vào đường dẫn đó.
  Nếu show_vis=True, mở ảnh bằng Image.show().
  """
  # 1. Load image
  resp = requests.get(image_url)
  resp.raise_for_status()
  img = Image.open(BytesIO(resp.content)).convert('RGB')

  # 2. Tạo full‐mask
  full_mask = Image.new("L", img.size, 255)

  # 4. Tạo prompt
  prompt = f"""<image>\nYou are a Visual Question Answering system.
                Given an image and a question, you must propose the top 5 candidate answers, each with a confidence score (from 0.00 to 1.00), sorted in descending order.
                – Base your answers only on clearly visible information in the image, without any external inference.
                – Output only one line in the following exact format:
                Candidates: answer1(score1), answer2(score2), answer3(score3), answer4(score4), answer5(score5)

                Example 1:
                Question: What color is the car in the image?
                Candidates: red(0.98), orange(0.75), yellow(0.40), brown(0.15), white(0.05)

                Example 2:
                Question: How many people are visible?
                Candidates: two(0.92), three(0.60), one(0.30), four(0.10), zero(0.02)

                Now apply to the new case:

                Question: {question.strip()}
                Answer:"""

  # 5. Chạy DAM và in kết quả
  print("Description:")
  result = dam.get_description(
      img,
      full_mask,
      prompt,
      streaming=False,
      temperature=0.2,
      top_p=0.5,
      num_beams=1,
      max_new_tokens=512
  )
  return result


# if __name__ == "__main__":
#     # VD dùng hàm từ dòng lệnh
#   URL = "https://github.com/NVlabs/describe-anything/blob/main/images/1.jpg?raw=true"
#   Q = "What color is the dog's fur?"
#   result = run_full_image_vqa(URL, Q)
#   print(result)
