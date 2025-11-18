import base64

with open("assets/Stock-sense logo(1).jpeg", "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()

print(encoded)
