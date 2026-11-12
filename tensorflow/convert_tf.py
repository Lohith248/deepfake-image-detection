"""ONNX -> TF SavedModel via onnx2tf. Usage: convert_tf.py <name>   (reads <name>_s.onnx, writes sm_<name>/)
onnx2tf tries to download a sample image for its internal self-check; that network fetch is replaced
by random data (our own equivalence check is check_equivalence.py)."""
import sys, numpy as np
import onnx2tf, onnx2tf.onnx2tf as m
m.download_test_image_data = lambda: np.random.RandomState(0).rand(20, 224, 224, 3).astype(np.float32)
n = sys.argv[1]
onnx2tf.convert(input_onnx_file_path=f'{n}_s.onnx', output_folder_path=f'sm_{n}', not_use_onnxsim=True, output_signaturedefs=True,
                copy_onnx_input_output_names_to_tflite=False, disable_strict_mode=False)
