import requests
from transformers import CLIPProcessor, CLIPModel

# Load the CLIP model and processor
model_name = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

# Define the categories
categories = ['Bag', 'Dress', 'Rug', 'Shoes', 'other']

def zero_shot_classification(image_url):
    # Load the image from the URL
    image = requests.get(image_url).content

    # Encode the image and category text
    inputs = processor(text=categories, images=image, return_tensors="pt", padding=True)

    # Perform zero-shot classification
    logits_per_image, logits_per_text = model(**inputs)
    probs = logits_per_image.softmax(1)  # We can take the softmax to get the label probabilities

    # Get the predicted category
    predicted_category = categories[probs.argmax()]

    return predicted_category

# Example usage:
image_url = "https://www.net-a-porter.com/variants/images/1647597294776703/in/w920_q60.jpg"
predicted_category = zero_shot_classification(image_url)
print("Predicted Category:", predicted_category)
