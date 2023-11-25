from transformers import pipeline
import pandas as pd
From Localmodel import ModelHassan
#checkpoint = "openai/clip-vit-large-patch14"
detector =ModelHassan
labels_for_classification =  ['Bag', 'Dress', 'Rug', 'Shoes',"other"]
def img_class(image_to_classify):
    scores = detector(image_to_classify, 
                        candidate_labels = labels_for_classification)
    return(scores[:2])

print(img_class("/home/hassan/mytindy/model/Data/Untitled.jpg"))
