# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/7/10 4:28 PM
# @File: clip_model_test
# @Email: mlshenkai@163.com
from PIL import Image
import requests
import torch
#
# from transformers import (
#     LlamaConfig,
#     LlamaModel,
#     LlamaForCausalLM,
#     CLIPVisionModel,
#     CLIPImageProcessor,
# )
# #
# # # processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
# model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
# model.requires_grad_(False)
# model = model.to(torch.bfloat16)
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
#
# processed_image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
# processed_image = processed_image.unsqueeze(0)
# print(processed_image.dim())
# print(processed_image.shape)
#
# print(processed_image)
# with torch.no_grad():
#     output = model(processed_image,output_hidden_states=True)
#     print(output)

def test_clip_version(processor, models):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    processed_image = processor(images=image, return_tensors="pt")["pixel_values"][0]
    processed_image = processed_image.unsqueeze(0)
    output = models(pixel_values=processed_image, output_hidden_states=True)
    print(output)
    return output
