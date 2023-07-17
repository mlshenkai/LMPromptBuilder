# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2023/7/10 7:44 PM
# @File: clip_test_2
# @Email: mlshenkai@163.com
from transformers import CLIPImageProcessor, CLIPVisionModel
from src.test.clip_model_test import test_clip_version

image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14')


vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14')
test_clip_version(processor=image_processor, models=vision_tower)