import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np
import argparse 


def write_graph(frozen_func, out_path, out_name):
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=out_path,
                  name=f"{out_name}.pb",
                  as_text=False)
# Save its text representation
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                    logdir=out_path,
                    name=f"{out_name}.pbtxt",
                    as_text=True)

def main():
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input', type=str, help='input model', required=True, dest='input')
    args.add_argument('-o', '--output', type=str, help='output model', required=True, dest='output')
    args.add_argument('-p', '--path', type=str, help='path to save', required=True, dest='path')

    args = args.parse_args()

    if not(args.input and args.output and args.path):
        print("Please provide input, output and path")
    if args.input and args.output and args.path:
        #path of the directory where you want to save your model
        frozen_out_path = args.path
        # name of the .pb file
        frozen_graph_filename = args.output
        model = load_model(args.input)

        # Convert Keras model to ConcreteFunction
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 60)
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)
        print("-" * 60)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)
        # Save frozen graph to disk
        write_graph(frozen_func, args.path, args.output)
    else:
        print("Please provide input, output and path")