import numpy as np
import pickle
from pathlib import Path
from openpilot.common.numpy_fast import interp
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.modeld.runners import ModelRunner, Runtime
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.models.commonmodel_pyx import ModelFrame, CLContext
import cv2
from cereal.visionipc import VisionBuf

# Load model metadata
METADATA_PATH = Path('/home/ruoyu/openpilot/selfdrive/modeld/models/supercombo_metadata.pkl')
with open(METADATA_PATH, 'rb') as f:
    model_metadata = pickle.load(f)

MODEL_PATHS = {
    ModelRunner.THNEED: Path('/home/ruoyu/openpilot/selfdrive/modeld/models/supercombo.thneed'),
    ModelRunner.ONNX: Path('/home/ruoyu/openpilot/selfdrive/modeld/models/supercombo.onnx')
}

# Initialize CLContext and ModelFrame
cl_context = CLContext()
frame = ModelFrame(cl_context)

# Initialize the model
output_slices = model_metadata['output_slices']
net_output_size = model_metadata['output_shapes']['outputs'][1]
output = np.zeros(net_output_size, dtype=np.float32)
model = ModelRunner(MODEL_PATHS, output, Runtime.GPU, False, cl_context)
model.addInput("input_imgs", None)
model.addInput("big_input_imgs", None)

# Prepare the inputs
# NOTE: Replace this with actual inputs as per the original code's requirements
inputs = {
    'desire': np.zeros(ModelConstants.DESIRE_LEN * (ModelConstants.HISTORY_BUFFER_LEN+1), dtype=np.float32),
    'traffic_convention': np.zeros(ModelConstants.TRAFFIC_CONVENTION_LEN, dtype=np.float32),
    # Add other inputs as required
}

for k, v in inputs.items():
    model.addInput(k, v)

def run_inference(input_image):
    # Prepare the image (replace this with actual image preparation logic)
    # Example: transform = get_warp_matrix(...)
    transform = np.eye(3, dtype=float)  # Placeholder transformation matrix

    model_transform_extra = np.zeros((3, 3), dtype=np.float32)
    # Set the input buffer
    model.setInputBuffer("input_imgs", frame.prepare(input_image, model_transform_extra.flatten(), model.getCLBuffer("input_imgs")))

    # Execute the model
    model.execute()
    parser = Parser()
    outputs = parser.parse_outputs({k: output[np.newaxis, v] for k, v in output_slices.items()})

    # Return the model's outputs
    return outputs

# Example usage
# image = ... # Load your image here
# model_outputs = run_inference(image)

if __name__ == "__main__":
    from PIL import Image
    image = Image.open("./figures/1.PNG")
    model_outputs = run_inference(image)
    print(model_outputs)