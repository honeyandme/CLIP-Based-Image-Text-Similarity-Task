from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel
import torch
from torch.nn.utils.rnn import pad_sequence
#
model = CLIPVisionModel.from_pretrained("../vit_clip_data")
processor = AutoProcessor.from_pretrained("../vit_clip_data")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output  # pooled CLS states
print(pooled_output.shape)
# sequences = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
# padded_sequences = pad_sequence(sequences, batch_first=True)
# print(padded_sequences)

