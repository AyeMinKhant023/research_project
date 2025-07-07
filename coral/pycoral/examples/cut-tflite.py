import tensorflow as tf
import subprocess
import os


def cut_and_convert_to_tflite_model(base_model: tf.keras.Model, cutlayer: str, output_folder: str) -> None:

    input_tensor = base_model.input

    if cutlayer is not None:
        x = base_model.get_layer(cutlayer).output
    else:
        x = base_model.output

    # Dont append pooling layer if the cut layer is already pooling layer
    # None means not cutting anything
    if cutlayer is not None and 'pool' not in cutlayer:
        # Add custom global average pooling
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Define the final model
    cut_model = tf.keras.Model(inputs=input_tensor, outputs=x)
    cut_model.summary()


    # Cut at global average pooling layer
    #cut_layer = base_model.get_layer("global_average_pooling2d")
    #cut_model = tf.keras.Model(inputs=base_model.input, outputs=cut_layer.output)

    #cut_model.summary()

    # Optional: test the cut model
    # dummy_input = tf.random.uniform([1, *base_model.input_shape[1:]])
    # features = cut_model(dummy_input)
    # print(features.shape)

    # Define representative dataset for quantization
    def representative_data_gen():
        for _ in range(100):
            data = tf.random.uniform([1, *base_model.input_shape[1:]], 0, 1)
            yield [tf.cast(data, tf.float32)]

    # Convert to fully quantized TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(cut_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT ]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                           tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
                                           tf.lite.OpsSet.SELECT_TF_OPS] # enable TensorFlow ops.]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.experimental_new_quantizer = True
    converter.experimental_new_converter = True


    tflite_model = converter.convert()

    out_tflite_path = os.path.join(output_folder, base_model.name+'_'+cutlayer+'.tflite')

    # Save the quantized TFLite model
    with open(out_tflite_path, "wb") as f:
        f.write(tflite_model)

    print("Saved: ", out_tflite_path)

    interpreter = tf.lite.Interpreter(model_path=out_tflite_path)
    interpreter.allocate_tensors()

    tensor_details = interpreter.get_tensor_details()

    for i, t in enumerate(tensor_details):
        print(f"[{i}] name: {t['name']}, shape: {t['shape']}, dtype: {t['dtype']}")


    # Converting it to edgetpu
    subprocess.run(["edgetpu_compiler", out_tflite_path, "-o", "./out"]) 


mobilenetv1 = dict()
mobilenetv1['model'] = tf.keras.applications.MobileNet(weights='imagenet', include_top=True)
mobilenetv1['cutlayer'] = ['conv_pw_'+str(i)+'_relu' for i in range(2,14)]
#mobilenetv1['cutlayer'] = ['global_average_pooling2d']

mobilenetv2 = dict()
mobilenetv2['model'] = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=True)
mobilenetv2['cutlayer'] = ['block_'+str(i)+'_project_BN' for i in range(2,17)]

# EfficientNetV2 doesnt seem to be edge tpu compatible
# We should download the cpu version, cut it and compile it to tpu instead
# https://research.google/blog/efficientnet-edgetpu-creating-accelerator-optimized-neural-networks-with-automl/
# https://coral.ai/models/image-classification/
if False:
    effnetS = dict()
    effnetS['model'] = tf.keras.applications.EfficientNetV2S(weights='imagenet', include_top=True, input_shape=(384, 384, 3),)
    effnetS['cutlayer'] = ['avg_pool']
    #effnetS['cutlayer'] = ['block6o_add']

    effnetM = dict()
    effnetM['model'] = tf.keras.applications.EfficientNetV2M(weights='imagenet', include_top=True)
    effnetM['cutlayer'] = ['avg_pool']

    effnetL = dict()
    effnetL['model'] = tf.keras.applications.EfficientNetV2L(weights='imagenet', include_top=True)
    effnetL['cutlayer'] = ['avg_pool']

inceptionv3 = dict()
inceptionv3['model'] = tf.keras.applications.InceptionV3(weights='imagenet', include_top=True)
inceptionv3['cutlayer'] = ['avg_pool'] # add more layers to cut if needed

resnet50 = dict()
resnet50['model'] = tf.keras.applications.ResNet50(weights='imagenet', include_top=True)
resnet50['cutlayer'] = ['avg_pool'] # add more layers to cut if needed

mobilenetv1['model'].summary()
mobilenetv2['model'].summary()
inceptionv3['model'].summary()
resnet50['model'].summary() 
exit(1)

experiments = [mobilenetv1, mobilenetv2, inceptionv3, resnet50]
experiments = [inceptionv3]
output_folder = './out/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for exp in experiments:

    exp['model'].summary()
    print(exp['model'].name, exp['model'].input_shape)


    for cutlayer in exp['cutlayer']:
        cut_and_convert_to_tflite_model(exp['model'], cutlayer, output_folder)