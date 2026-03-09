import predict
# Sample input data (4 features for the Iris dataset)
sample_input = ['5.1', '3.5', '1.4', '0.2']
def test_predict():
   
    # Call the predict function
    prediction = predict.predict(sample_input)
    
    # Check if the prediction is a valid class label (0, 1, or 2 for Iris dataset)
    assert prediction in [0, 1, 2], f"Unexpected prediction: {prediction}"  


# now run the test
if __name__ == "__main__":
    test_predict()
    print("Test passed!")  
    print (f"Prediction for sample input {sample_input}: {predict.predict(sample_input)}")     