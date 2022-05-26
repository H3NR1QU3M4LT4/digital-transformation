import pickle
import numpy as np

def predicitons(path_model, data):
    aux_dict = {}
    
    model = pickle.load(open(path_model, "rb"))

    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(list(data.values()))
    numeric_array = np.asarray(input_data_as_numpy_array, dtype=float)
    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = numeric_array.reshape(1, -1)

    prediction = model.predict(input_data_reshaped)
        
    class_probabilities = model.predict_proba(input_data_reshaped)
    confidence = max(list(class_probabilities[0]))
        
    aux_dict['prediction'] = int(prediction[0])
    aux_dict['confidence'] = confidence
    
    return aux_dict