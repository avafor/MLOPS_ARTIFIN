
def validate_input(features):
    if len(features) != 4:
        raise ValueError("Iris input must contain exactly 4 features")

    for x in features:
        if not isinstance(x, (int, float)):
            raise ValueError("All features must be numeric")


def predict(features, model):
    validate_input(features)
    return model.predict([features])