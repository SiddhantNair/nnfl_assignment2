from inception_v3 import InceptionV3, image_from_file, get_label

if __name__ == "__main__":

    # Load Inception V3 Model trained on imagenet
    model = InceptionV3()

    # Load Image for Classification
    image = image_from_file('meow.jpeg')

    # Model Features for Classification
    possibilities = model.predict(image)

    # Classify image
    prediction = get_label(possibilities)
    print(prediction)
