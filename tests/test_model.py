from ml_model.train import train_model
from ml_model.predict import predict

def test_train_model():
    model = train_model()
    assert model is not None

def test_predict():
    sample_input = [[3, 3], [4, 4]]
    prediction = predict(sample_input)
    assert prediction is not None
    assert len(prediction) == 2

    
