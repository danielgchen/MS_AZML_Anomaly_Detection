import json, os
import numpy as np
import onnxruntime as rt


def init():
    '''
    setup the ONNX model
    '''
    global session, input_name, label_name
    # define the model path
    # TODO: we need to remove <model name> (i.e. isolation_forest/) for triton?
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'models/triton/isolation_forest/1/isolation_forest.onnx')
    # deserialize the model file back into a sklearn model
    session = rt.InferenceSession(model_path)
    # get session informaiton
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    print(f'Input name: {input_name}')
    print(f'Label name: {label_name}')

def run(data):
    '''
    run inference on input data
    '''
    # TODO: right now we can only handle one data what if we want to handle multiple?
    # read in data
    data = json.loads(data)
    # preprocess it to transform into input
    data = preprocess(data['data'])  # 'data' is arbitary we need to make it loopable
    # predict output
    try:
        result = session.run([label_name], {input_name: data})
        return(result[0].tolist())  # the [0] is only for one data entry
    except Exception as error:
        return(str(error))
    
def preprocess(data):
    '''
    takes in list of lists input data and gives an ONNX `float` so float32 numpy array
    '''
    # TODO: change it so it considers float64 aka doubles?
    data = np.array(data).astype('float32')
    return(data)


