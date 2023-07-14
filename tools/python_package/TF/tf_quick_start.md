# Layers
## Dense全连接层


```python
import tensorflow as tf

# 定义一个包含3个神经元的Dense层
dense_layer = tf.keras.layers.Dense(
                units=3, 
                activation='relu', # 默认None
                name="layer1", 
                kernel_initializer='glorot_uniform', # 默认
                bias_initializer='zeros', # 默认
                )

# 假设有一个输入张量x
x = tf.ones((1, 10)) # 输入形状为(1, 10)

# 将x传递给dense_layer
y = dense_layer(x)

# 输出y的形状
print(y.shape) # 输出 (1, 3)
print(y)
```

    (1, 3)
    tf.Tensor([[1.3682876 0.        1.2488521]], shape=(1, 3), dtype=float32)



Dense层的一些重要参数包括：
- **units**: 输出空间的维度，即该层有多少个神经元。
- **activation**: 激活函数。默认为线性激活函数None。常用的激活函数包括relu、sigmoid、tanh等。
- **kernel_initializer**: 权重矩阵初始化器，默认为Glorot正态分布初始化器，也称为 Xavier 正态分布初始化器。它从以 0 为中心，标准差为 stddev = sqrt(2 / (fan_in + fan_out)) 的截断正态分布中抽取样本， 其中 fan_in 是权值张量中的输入单位的数量， fan_out 是权值张量中的输出单位的数量
- **bias_initializer**: 偏置向量的初始化器，默认为0值初始化
- Dense层还有许多其他的参数，如kernel_regularizer等，用于控制权重矩阵正则化方式



在上面的例子中:
- 我们首先定义了一个包含3个神经元的Dense层，然后创建了一个形状为(1, 10)的输入张量x，我们将x传递给dense_layer，它对输入进行加权求和和非线性变换，并输出形状为(1, 3)的张量y。
- 在Dense层中，我们需要将输入x和权重矩阵进行矩阵乘法，并加上偏置向量，然后再将结果传递给激活函数进行非线性变换。计算过程为$output = activation(dot(input, kernel) + bias)$
- **input**：表示输入张量，形状为(batch_size, input_dim)，这里为(1，10)，其中batch_size是输入数据的批量大小，input_dim是输入数据的维度，即每个输入样本的特征数量
- **kernel**：表示权重矩阵，形状为(input_dim, units)，这里为(10，3)，input_dim是输入数据的每个输入样本的特征数量，units为自定义的神经元个数
- **bias**：表示偏置向量，形状为(units，)，这里为(3，)，units为自定义的神经元个数
- **activation**：表示激活函数，这里采用Relu，即Rectified Linear Unit激活函数，它对输入进行非线性变换，将小于0的值截断为0

如果输入张量x阶数大于2，那么计算输入x和权重矩阵会按照最后一维计算，比如x(3,5,7),kernel(7,3),输出(3,5,3)


```python
# 矩阵乘法，得到形状为(batch_size, 3)的结果
output = tf.matmul(x_batch, kernel)

# 加上偏置向量，得到形状为(batch_size, 3)的结果
output = tf.nn.bias_add(output, bias)

# 进行非线性变换，得到形状为(batch_size, 3)的结果
output = tf.nn.relu(output)
```

#  顺序模型


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

## 何时使用顺序模型
模型Sequential适用于简单的层堆栈 ，其中每一层都有一个输入张量和一个输出张量。




```python
# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)
y
```




    <tf.Tensor: shape=(3, 4), dtype=float32, numpy=
    array([[ 0.03704125, -0.6245982 , -0.12151273,  0.3119151 ],
           [ 0.03704125, -0.6245982 , -0.12151273,  0.3119151 ],
           [ 0.03704125, -0.6245982 , -0.12151273,  0.3119151 ]],
          dtype=float32)>



相当于这个函数：


```python
# Create 3 layers
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")

# Call layers on a test input
x = tf.ones((3, 3))
y = layer3(layer2(layer1(x)))
y
```




    <tf.Tensor: shape=(3, 4), dtype=float32, numpy=
    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]], dtype=float32)>



在以下情况下，顺序模型不适用：
- 您的模型有多个输入或多个输出
- 您的任何层都有多个输入或多个输出
- 您需要进行图层共享
- 您需要非线性拓扑（例如，残差连接、多分支模型）

## 创建顺序模型
您可以通过将层列表传递给 Sequential 构造函数来创建 Sequential 模型


```python
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)
```

它的层可以通过属性访问layers：


```python
model.layers
```




    [<keras.layers.core.dense.Dense at 0x7fa909c46310>,
     <keras.layers.core.dense.Dense at 0x7fa909c46af0>,
     <keras.layers.core.dense.Dense at 0x7fa909c857c0>]



您还可以通过以下方法逐步创建顺序模型add()：


```python
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))
```

请注意，还有一种相应的pop()方法来删除层：顺序模型的行为非常类似于层列表


```python
model.pop()
print(len(model.layers))  # 2
```

    2


另请注意，Sequential 构造函数接受一个name参数，就像 Keras 中的任何层或模型一样。这对于使用具有语义意义的名称注释 TensorBoard 图很有用。


```python
model = keras.Sequential(name="my_sequential")
model.add(layers.Dense(2, activation="relu", name="layer1"))
model.add(layers.Dense(3, activation="relu", name="layer2"))
model.add(layers.Dense(4, name="layer3"))
```

## 预先指定层输入形状

通常，Keras 中的所有层都需要知道其输入的形状，以便能够创建它们的权重。因此，当您创建这样的层时，最初它没有权重：


```python
layer = layers.Dense(3)
layer.weights  # Empty
```




    []



它在第一次调用输入时创建权重，因为权重的形状取决于输入的形状：


```python
# Call layer on a test input
x = tf.ones((1, 4))
y = layer(x)
layer.weights  # Now it has weights, of shape (4, 3) and (3,)
```




    [<tf.Variable 'dense_8/kernel:0' shape=(4, 3) dtype=float32, numpy=
     array([[ 0.77282166, -0.6165689 , -0.01775068],
            [-0.77349555, -0.6670578 , -0.19441098],
            [ 0.7029178 ,  0.7533556 ,  0.21160185],
            [-0.51589584, -0.82704777,  0.8298104 ]], dtype=float32)>,
     <tf.Variable 'dense_8/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>]



当然，这也适用于顺序模型。当您在没有输入形状的情况下实例化一个顺序模型时，它不是“构建的”：它没有权重（并且调用会导致 model.weights错误）。权重是在模型第一次看到一些输入数据时创建的：


```python
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ]
)  # No weights at this stage!

# At this point, you can't do this:
# model.weights

# You also can't do this:
# model.summary()

# Call the model on a test input
x = tf.ones((1, 4))
y = model(x)
print("Number of weights after calling the model:", len(model.weights))  # 6
```

    Number of weights after calling the model: 6


“构建”模型后，您可以调用其summary()方法来显示其内容：


```python
model.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_9 (Dense)             (1, 2)                    10        
                                                                     
     dense_10 (Dense)            (1, 3)                    9         
                                                                     
     dense_11 (Dense)            (1, 4)                    16        
                                                                     
    =================================================================
    Total params: 35
    Trainable params: 35
    Non-trainable params: 0
    _________________________________________________________________


但是，在以增量方式构建顺序模型以能够显示到目前为止的模型摘要（包括当前输出形状）时，它可能非常有用。在这种情况下，您应该通过将一个对象传递给您的模型来启动您的模型Input ，以便它从一开始就知道它的输入形状：


```python
model = keras.Sequential()
model.add(keras.Input(shape=(4,)))
model.add(layers.Dense(2, activation="relu"))

model.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_12 (Dense)            (None, 2)                 10        
                                                                     
    =================================================================
    Total params: 10
    Trainable params: 10
    Non-trainable params: 0
    _________________________________________________________________


请注意，该Input对象不显示为 的一部分model.layers，因为它不是图层：


```python
model.layers
```




    [<keras.layers.core.dense.Dense at 0x7fa909c9a820>]



一个简单的替代方法是将input_shape参数传递给第一层：


```python
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu", input_shape=(4,)))

model.summary()

```

    Model: "sequential_5"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_13 (Dense)            (None, 2)                 10        
                                                                     
    =================================================================
    Total params: 10
    Trainable params: 10
    Non-trainable params: 0
    _________________________________________________________________


使用像这样的预定义输入形状构建的模型始终具有权重（甚至在看到任何数据之前）并且始终具有定义的输出形状。

通常，如果您知道序列模型是什么，建议始终提前指定其输入形状。


```python
model.weights
```




    [<tf.Variable 'dense_13/kernel:0' shape=(4, 2) dtype=float32, numpy=
     array([[ 0.5970671 ,  0.15860963],
            [-0.8598833 , -0.3841095 ],
            [ 0.11947203,  0.68715477],
            [-0.46637487, -0.6774483 ]], dtype=float32)>,
     <tf.Variable 'dense_13/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>]



## 一个常见的调试工作流程：add()+summary()
在构建新的顺序架构时，使用增量堆叠层并add()频繁打印模型摘要非常有用。例如，这使您能够监视堆栈Conv2D和MaxPooling2D图层如何对图像特征图进行下采样：


```python
model = keras.Sequential()
model.add(keras.Input(shape=(250, 250, 3)))  # 250x250 RGB images
model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))

# Can you guess what the current output shape is at this point? Probably not.
# Let's just print it:
model.summary()

# The answer was: (40, 40, 32), so we can keep downsampling...

model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(2))

# And now?
model.summary()

# Now that we have 4x4 feature maps, time to apply global max pooling.
model.add(layers.GlobalMaxPooling2D())

# Finally, we add a classification layer.
model.add(layers.Dense(10))
```

    Model: "sequential_6"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 123, 123, 32)      2432      
                                                                     
     conv2d_1 (Conv2D)           (None, 121, 121, 32)      9248      
                                                                     
     max_pooling2d (MaxPooling2D  (None, 40, 40, 32)       0         
     )                                                               
                                                                     
    =================================================================
    Total params: 11,680
    Trainable params: 11,680
    Non-trainable params: 0
    _________________________________________________________________
    Model: "sequential_6"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 123, 123, 32)      2432      
                                                                     
     conv2d_1 (Conv2D)           (None, 121, 121, 32)      9248      
                                                                     
     max_pooling2d (MaxPooling2D  (None, 40, 40, 32)       0         
     )                                                               
                                                                     
     conv2d_2 (Conv2D)           (None, 38, 38, 32)        9248      
                                                                     
     conv2d_3 (Conv2D)           (None, 36, 36, 32)        9248      
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 12, 12, 32)       0         
     2D)                                                             
                                                                     
     conv2d_4 (Conv2D)           (None, 10, 10, 32)        9248      
                                                                     
     conv2d_5 (Conv2D)           (None, 8, 8, 32)          9248      
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 4, 4, 32)         0         
     2D)                                                             
                                                                     
    =================================================================
    Total params: 48,672
    Trainable params: 48,672
    Non-trainable params: 0
    _________________________________________________________________


## 拥有模型后该做什么

一旦您的模型架构准备就绪，您将需要：
- 训练您的模型、对其进行评估并运行推理。请参阅我们的 [内置训练方法和评估指南](https://www.tensorflow.org/guide/keras/train_and_evaluate?hl=zh-cn)
- 将您的模型保存到磁盘并恢复它。请参阅我们的 [序列化和保存指南](https://www.tensorflow.org/guide/keras/save_and_serialize?hl=zh-cn)。
- 利用多个 GPU 加速模型训练。请参阅我们的[多GPU和分布式训练指南](https://keras.io/guides/distributed_training/)。

## 使用顺序模型进行特征提取
一旦构建了 Sequential 模型，它的行为就像一个Functional API model。这意味着每一层都有一个input 和output属性。这些属性可以用来做一些整洁的事情，比如快速创建一个模型来提取顺序模型中所有中间层的输出


```python
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=[layer.output for layer in initial_model.layers],
)

# Call feature extractor on test input.
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)
```

这是一个类似的例子，它只从某一层中提取特征：


```python
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=initial_model.get_layer(name="my_intermediate_layer").output,
)
# Call feature extractor on test input.
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)
```

## 使用顺序模型进行迁移学习

迁移学习包括冻结模型中的底层并仅训练顶层。如果您不熟悉它，请务必阅读我们的[迁移学习指南](https://www.tensorflow.org/guide/keras/transfer_learning?hl=zh-cn)。

这是涉及顺序模型的两个常见迁移学习蓝图。

首先，假设您有一个顺序模型，并且您想要冻结除最后一层之外的所有层。在这种情况下，您只需遍历 model.layers并设置layer.trainable = False除最后一层之外的每一层。像这样：


```python
model = keras.Sequential([
    keras.Input(shape=(784)),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10),
])

# Presumably you would want to first load pre-trained weights.
model.load_weights(...)

# Freeze all layers except the last one.
for layer in model.layers[:-1]:
  layer.trainable = False

# Recompile and train (this will only update the weights of the last layer).
model.compile(...)
model.fit(...)
```

另一个常见的蓝图是使用顺序模型来堆叠预训练模型和一些新初始化的分类层。像这样


```python
# Load a convolutional base with pre-trained weights
base_model = keras.applications.Xception(
    weights='imagenet',
    include_top=False,
    pooling='avg')

# Freeze the base model
base_model.trainable = False

# Use a Sequential model to add a trainable classifier on top
model = keras.Sequential([
    base_model,
    layers.Dense(1000),
])

# Compile & train
model.compile(...)
model.fit(...)
```

如果你进行迁移学习，你可能会发现自己经常使用这两种模式。

这就是您需要了解的有关顺序模型的所有信息！

# 函数式API


```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

## 简介
Keras 函数式 API 是一种比 tf.keras.Sequential API 更加灵活的模型创建方式。函数式 API 可以处理具有非线性拓扑的模型、具有共享层的模型，以及具有多个输入或输出的模型。

深度学习模型通常是层的有向无环图 (DAG)。因此，函数式 API 是构建层计算图的一种方式。

请考虑以下模型：
- (input: 784-dimensional vectors) 
- [Dense (64 units, relu activation)]
- [Dense (64 units, relu activation)] 
- [Dense (10 units, softmax activation)] 
- (output: logits of a probability distribution over 10 classes)

这是一个具有三层的基本计算图。要使用函数式 API 构建此模型，请先创建一个输入节点：


```python
inputs = keras.Input(shape=(784,))
```

数据的形状设置为 784 维向量。由于仅指定了每个样本的形状，因此始终忽略批次大小。

例如，如果您有一个形状为 (32, 32, 3) 的图像输入，则可以使用：


```python
# Just for demonstration purposes.
img_inputs = keras.Input(shape=(32, 32, 3))
```

返回的 inputs 包含馈送给模型的输入数据的形状和 dtype。形状如下：


```python
inputs.shape
```




    TensorShape([None, 784])




```python
img_inputs.shape
```




    TensorShape([None, 32, 32, 3])



dtype 如下：


```python
inputs.dtype
```




    tf.float32



可以通过在此 inputs 对象上调用层，在层计算图中创建新的节点：



```python
dense = layers.Dense(64, activation="relu")
x = dense(inputs)
```

“层调用”操作就像从“输入”向您创建的该层绘制一个箭头。您将输入“传递”到 dense 层，然后得到 x。

让我们为层计算图多添加几个层：


```python
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)
```

定义模型输入输出和名字


```python
model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
```

让我们看看模型摘要是什么样子：


```python
model.summary()
```

    Model: "mnist_model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_1 (InputLayer)        [(None, 784)]             0         
                                                                     
     dense (Dense)               (None, 64)                50240     
                                                                     
     dense_1 (Dense)             (None, 64)                4160      
                                                                     
     dense_2 (Dense)             (None, 10)                650       
                                                                     
    =================================================================
    Total params: 55,050
    Trainable params: 55,050
    Non-trainable params: 0
    _________________________________________________________________


您还可以将模型绘制为计算图：


```python
keras.utils.plot_model(model, "my_first_model.png")
```




    
![png](t5_files/t5_72_0.png)
    



并且，您还可以选择在绘制的计算图中显示每层的输入和输出形状：


```python
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
```




    
![png](t5_files/t5_74_0.png)
    



此图和代码几乎完全相同。在代码版本中，连接箭头由调用操作代替。

“层计算图”是深度学习模型的直观心理图像，而函数式 API 是创建密切反映此图像的模型的方法。

## 训练，评估和推断
对于使用函数式 API 构建的模型来说，其训练、评估和推断的工作方式与 Sequential 模型完全相同。

Model 类提供了一个内置训练循环（fit() 方法）和一个内置评估循环（evaluate() 方法）。请注意，您可以轻松地[自定义这些循环](https://tensorflow.google.cn/guide/keras/customizing_what_happens_in_fit/?hl=zh-cn)，以实现监督学习之外的训练例程（例如 GAN）。

如下所示，加载 MNIST 图像数据，将其改造为向量，将模型与数据拟合（同时监视验证拆分的性能），然后在测试数据上评估模型：


```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
```

    Epoch 1/2
    750/750 [==============================] - 2s 2ms/step - loss: 0.3499 - accuracy: 0.9002 - val_loss: 0.1947 - val_accuracy: 0.9436
    Epoch 2/2
    750/750 [==============================] - 1s 1ms/step - loss: 0.1593 - accuracy: 0.9521 - val_loss: 0.1611 - val_accuracy: 0.9523
    313/313 - 0s - loss: 0.1671 - accuracy: 0.9478 - 227ms/epoch - 725us/step
    Test loss: 0.16706955432891846
    Test accuracy: 0.9477999806404114


有关更多信息，请参阅训练和评估指南。
## 保存和序列化
对于使用函数式 API 构建的模型，其保存模型和序列化的工作方式与 Sequential 模型相同。保存函数式模型的标准方式是调用 model.save() 将整个模型保存为单个文件。您可以稍后从该文件重新创建相同的模型，即使构建该模型的代码已不再可用。

保存的文件包括：
- 模型架构
- 模型权重值（在训练过程中得知）
- 模型训练配置（如果有的话，如传递给 compile）
- 优化器及其状态（如果有的话，用来从上次中断的地方重新开始训练）


```python
model.save("path_to_my_model")
del model
# Recreate the exact same model purely from the file:
model = keras.models.load_model("path_to_my_model")
```

    WARNING:absl:Found untraced functions such as _update_step_xla while saving (showing 1 of 1). These functions will not be directly callable after loading.


    INFO:tensorflow:Assets written to: path_to_my_model/assets


    INFO:tensorflow:Assets written to: path_to_my_model/assets


有关详细信息，请阅读模型[序列化和保存指南](https://tensorflow.google.cn/guide/keras/save_and_serialize/?hl=zh-cn)。

## 所有模型均可像层一样调用1
在函数式 API 中，模型是通过在层计算图中指定其输入和输出来创建的。这意味着可以使用单个层计算图来生成多个模型。

在下面的示例中，您将使用相同的层堆栈来实例化两个模型：能够将图像输入转换为 16 维向量的 encoder 模型，以及用于训练的端到端 autoencoder 模型。


```python
encoder_input = keras.Input(shape=(28, 28, 1), name="img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

x = layers.Reshape((4, 4, 1))(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
autoencoder.summary()
```

    Model: "encoder"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     img (InputLayer)            [(None, 28, 28, 1)]       0         
                                                                     
     conv2d (Conv2D)             (None, 26, 26, 16)        160       
                                                                     
     conv2d_1 (Conv2D)           (None, 24, 24, 32)        4640      
                                                                     
     max_pooling2d (MaxPooling2D  (None, 8, 8, 32)         0         
     )                                                               
                                                                     
     conv2d_2 (Conv2D)           (None, 6, 6, 32)          9248      
                                                                     
     conv2d_3 (Conv2D)           (None, 4, 4, 16)          4624      
                                                                     
     global_max_pooling2d (Globa  (None, 16)               0         
     lMaxPooling2D)                                                  
                                                                     
    =================================================================
    Total params: 18,672
    Trainable params: 18,672
    Non-trainable params: 0
    _________________________________________________________________
    Model: "autoencoder"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     img (InputLayer)            [(None, 28, 28, 1)]       0         
                                                                     
     conv2d (Conv2D)             (None, 26, 26, 16)        160       
                                                                     
     conv2d_1 (Conv2D)           (None, 24, 24, 32)        4640      
                                                                     
     max_pooling2d (MaxPooling2D  (None, 8, 8, 32)         0         
     )                                                               
                                                                     
     conv2d_2 (Conv2D)           (None, 6, 6, 32)          9248      
                                                                     
     conv2d_3 (Conv2D)           (None, 4, 4, 16)          4624      
                                                                     
     global_max_pooling2d (Globa  (None, 16)               0         
     lMaxPooling2D)                                                  
                                                                     
     reshape (Reshape)           (None, 4, 4, 1)           0         
                                                                     
     conv2d_transpose (Conv2DTra  (None, 6, 6, 16)         160       
     nspose)                                                         
                                                                     
     conv2d_transpose_1 (Conv2DT  (None, 8, 8, 32)         4640      
     ranspose)                                                       
                                                                     
     up_sampling2d (UpSampling2D  (None, 24, 24, 32)       0         
     )                                                               
                                                                     
     conv2d_transpose_2 (Conv2DT  (None, 26, 26, 16)       4624      
     ranspose)                                                       
                                                                     
     conv2d_transpose_3 (Conv2DT  (None, 28, 28, 1)        145       
     ranspose)                                                       
                                                                     
    =================================================================
    Total params: 28,241
    Trainable params: 28,241
    Non-trainable params: 0
    _________________________________________________________________


在上例中，解码架构与编码架构严格对称，因此输出形状与输入形状 (28, 28, 1) 相同。

Conv2D 层的反面是 Conv2DTranspose 层，MaxPooling2D 层的反面是 UpSampling2D 层。

## 所有模型均可像层一样调用2
您可以通过在 Input 上或在另一个层的输出上调用任何模型来将其当作层来处理。通过调用模型，您不仅可以重用模型的架构，还可以重用它的权重。

为了查看实际运行情况，下面是对自动编码器示例的另一种处理方式，该示例创建了一个编码器模型、一个解码器模型，并在两个调用中将它们链接，以获得自动编码器模型：


```python
encoder_input = keras.Input(shape=(28, 28, 1), name="original_img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

decoder_input = keras.Input(shape=(16,), name="encoded_img")
x = layers.Reshape((4, 4, 1))(decoder_input)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

decoder = keras.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()

autoencoder_input = keras.Input(shape=(28, 28, 1), name="img")
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input, decoded_img, name="autoencoder")
autoencoder.summary()
```

    Model: "encoder"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     original_img (InputLayer)   [(None, 28, 28, 1)]       0         
                                                                     
     conv2d_4 (Conv2D)           (None, 26, 26, 16)        160       
                                                                     
     conv2d_5 (Conv2D)           (None, 24, 24, 32)        4640      
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 8, 8, 32)         0         
     2D)                                                             
                                                                     
     conv2d_6 (Conv2D)           (None, 6, 6, 32)          9248      
                                                                     
     conv2d_7 (Conv2D)           (None, 4, 4, 16)          4624      
                                                                     
     global_max_pooling2d_1 (Glo  (None, 16)               0         
     balMaxPooling2D)                                                
                                                                     
    =================================================================
    Total params: 18,672
    Trainable params: 18,672
    Non-trainable params: 0
    _________________________________________________________________
    Model: "decoder"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     encoded_img (InputLayer)    [(None, 16)]              0         
                                                                     
     reshape_1 (Reshape)         (None, 4, 4, 1)           0         
                                                                     
     conv2d_transpose_4 (Conv2DT  (None, 6, 6, 16)         160       
     ranspose)                                                       
                                                                     
     conv2d_transpose_5 (Conv2DT  (None, 8, 8, 32)         4640      
     ranspose)                                                       
                                                                     
     up_sampling2d_1 (UpSampling  (None, 24, 24, 32)       0         
     2D)                                                             
                                                                     
     conv2d_transpose_6 (Conv2DT  (None, 26, 26, 16)       4624      
     ranspose)                                                       
                                                                     
     conv2d_transpose_7 (Conv2DT  (None, 28, 28, 1)        145       
     ranspose)                                                       
                                                                     
    =================================================================
    Total params: 9,569
    Trainable params: 9,569
    Non-trainable params: 0
    _________________________________________________________________
    Model: "autoencoder"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     img (InputLayer)            [(None, 28, 28, 1)]       0         
                                                                     
     encoder (Functional)        (None, 16)                18672     
                                                                     
     decoder (Functional)        (None, 28, 28, 1)         9569      
                                                                     
    =================================================================
    Total params: 28,241
    Trainable params: 28,241
    Non-trainable params: 0
    _________________________________________________________________


如您所见，模型可以嵌套：模型可以包含子模型（因为模型就像层一样）。模型嵌套的一个常见用例是装配。例如，以下展示了如何将一组模型装配成一个平均其预测的模型：


```python
def get_model():
    inputs = keras.Input(shape=(128,))
    outputs = layers.Dense(1)(inputs)
    return keras.Model(inputs, outputs)


model1 = get_model()
model2 = get_model()
model3 = get_model()

inputs = keras.Input(shape=(128,))
y1 = model1(inputs)
y2 = model2(inputs)
y3 = model3(inputs)
outputs = layers.average([y1, y2, y3])
ensemble_model = keras.Model(inputs=inputs, outputs=outputs)
```

## 处理复杂的计算图拓扑
### 具有多个输入和输出的模型

函数式 API 使处理多个输入和输出变得容易。而这无法使用 Sequential API 处理。

例如，如果您要构建一个系统，该系统按照优先级对自定义问题工单进行排序，然后将工单传送到正确的部门，则此模型将具有三个输入：

- 工单标题（文本输入），
- 工单的文本正文（文本输入），以及
- 用户添加的任何标签（分类输入）

此模型将具有两个输出：

- 介于 0 和 1 之间的优先级分数（标量 Sigmoid 输出），以及
- 应该处理工单的部门（部门范围内的 Softmax 输出）。
- 您可以使用函数式 API 通过几行代码构建此模型：





```python
num_tags = 12  # Number of unique issue tags
num_words = 10000  # Size of vocabulary obtained when preprocessing text data
num_departments = 4  # Number of departments for predictions

title_input = keras.Input(
    shape=(None,), name="title"
)  # Variable-length sequence of ints
body_input = keras.Input(shape=(None,), name="body")  # Variable-length sequence of ints
tags_input = keras.Input(
    shape=(num_tags,), name="tags"
)  # Binary vectors of size `num_tags`

# Embed each word in the title into a 64-dimensional vector
title_features = layers.Embedding(num_words, 64)(title_input)
# Embed each word in the text into a 64-dimensional vector
body_features = layers.Embedding(num_words, 64)(body_input)

# Reduce sequence of embedded words in the title into a single 128-dimensional vector
title_features = layers.LSTM(128)(title_features)
# Reduce sequence of embedded words in the body into a single 32-dimensional vector
body_features = layers.LSTM(32)(body_features)

# Merge all available features into a single large vector via concatenation
x = layers.concatenate([title_features, body_features, tags_input])

# Stick a logistic regression for priority prediction on top of the features
priority_pred = layers.Dense(1, name="priority")(x)
# Stick a department classifier on top of the features
department_pred = layers.Dense(num_departments, name="department")(x)

# Instantiate an end-to-end model predicting both priority and department
model = keras.Model(
    inputs=[title_input, body_input, tags_input],
    outputs=[priority_pred, department_pred],
)
```

现在绘制模型


```python
keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
```




    
![png](t5_files/t5_91_0.png)
    



编译此模型时，可以为每个输出分配不同的损失。甚至可以为每个损失分配不同的权重，以调整其对总训练损失的贡献。


```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[
        keras.losses.BinaryCrossentropy(from_logits=True),
        keras.losses.CategoricalCrossentropy(from_logits=True),
    ],
    loss_weights=[1.0, 0.2],
)
```

由于输出层具有不同的名称，您还可以使用对应的层名称指定损失和损失权重：


```python
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={
        "priority": keras.losses.BinaryCrossentropy(from_logits=True),
        "department": keras.losses.CategoricalCrossentropy(from_logits=True),
    },
    loss_weights={"priority": 1.0, "department": 0.2},
)
```

通过传递输入和目标的 NumPy 数组列表来训练模型：


```python
# Dummy input data
title_data = np.random.randint(num_words, size=(1280, 10))
body_data = np.random.randint(num_words, size=(1280, 100))
tags_data = np.random.randint(2, size=(1280, num_tags)).astype("float32")

# Dummy target data
priority_targets = np.random.random(size=(1280, 1))
dept_targets = np.random.randint(2, size=(1280, num_departments))

model.fit(
    {"title": title_data, "body": body_data, "tags": tags_data},
    {"priority": priority_targets, "department": dept_targets},
    epochs=2,
    batch_size=32,
)
```

    Epoch 1/2
    40/40 [==============================] - 3s 18ms/step - loss: 1.2708 - priority_loss: 0.7099 - department_loss: 2.8045
    Epoch 2/2
    40/40 [==============================] - 1s 18ms/step - loss: 1.2666 - priority_loss: 0.7020 - department_loss: 2.8228





    <keras.callbacks.History at 0x7ff6013c01c0>



当使用 Dataset 对象调用fit时，它应该会生成一个列表元组（如 ([title_data, body_data, tags_data], [priority_targets, dept_targets]) 或一个字典元组（如 ({'title': title_data, 'body': body_data, 'tags': tags_data}, {'priority': priority_targets, 'department': dept_targets})）。

有关详细说明，请参阅训练和评估指南。
### 小 ResNet 模型
除了具有多个输入和输出的模型外，函数式 API 还使处理非线性连接拓扑（这些模型的层没有按顺序连接）变得容易。这是 Sequential API 无法处理的。

关于这一点的一个常见用例是残差连接。让我们来为 CIFAR10 构建一个小 ResNet 模型以进行演示：





```python
inputs = keras.Input(shape=(32, 32, 3), name="img")
x = layers.Conv2D(32, 3, activation="relu")(inputs)
x = layers.Conv2D(64, 3, activation="relu")(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs, outputs, name="toy_resnet")
model.summary()
```

    Model: "toy_resnet"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     img (InputLayer)               [(None, 32, 32, 3)]  0           []                               
                                                                                                      
     conv2d_8 (Conv2D)              (None, 30, 30, 32)   896         ['img[0][0]']                    
                                                                                                      
     conv2d_9 (Conv2D)              (None, 28, 28, 64)   18496       ['conv2d_8[0][0]']               
                                                                                                      
     max_pooling2d_2 (MaxPooling2D)  (None, 9, 9, 64)    0           ['conv2d_9[0][0]']               
                                                                                                      
     conv2d_10 (Conv2D)             (None, 9, 9, 64)     36928       ['max_pooling2d_2[0][0]']        
                                                                                                      
     conv2d_11 (Conv2D)             (None, 9, 9, 64)     36928       ['conv2d_10[0][0]']              
                                                                                                      
     add (Add)                      (None, 9, 9, 64)     0           ['conv2d_11[0][0]',              
                                                                      'max_pooling2d_2[0][0]']        
                                                                                                      
     conv2d_12 (Conv2D)             (None, 9, 9, 64)     36928       ['add[0][0]']                    
                                                                                                      
     conv2d_13 (Conv2D)             (None, 9, 9, 64)     36928       ['conv2d_12[0][0]']              
                                                                                                      
     add_1 (Add)                    (None, 9, 9, 64)     0           ['conv2d_13[0][0]',              
                                                                      'add[0][0]']                    
                                                                                                      
     conv2d_14 (Conv2D)             (None, 7, 7, 64)     36928       ['add_1[0][0]']                  
                                                                                                      
     global_average_pooling2d (Glob  (None, 64)          0           ['conv2d_14[0][0]']              
     alAveragePooling2D)                                                                              
                                                                                                      
     dense_6 (Dense)                (None, 256)          16640       ['global_average_pooling2d[0][0]'
                                                                     ]                                
                                                                                                      
     dropout (Dropout)              (None, 256)          0           ['dense_6[0][0]']                
                                                                                                      
     dense_7 (Dense)                (None, 10)           2570        ['dropout[0][0]']                
                                                                                                      
    ==================================================================================================
    Total params: 223,242
    Trainable params: 223,242
    Non-trainable params: 0
    __________________________________________________________________________________________________


绘制模型：


```python
keras.utils.plot_model(model, "mini_resnet.png", show_shapes=True)
```




    
![png](t5_files/t5_101_0.png)
    



现在训练模型：


```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["acc"],
)
# We restrict the data to the first 1000 samples so as to limit execution time
# on Colab. Try to train on the entire dataset until convergence!
model.fit(x_train[:1000], y_train[:1000], batch_size=64, epochs=1, validation_split=0.2)
```

    Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    170498071/170498071 [==============================] - 25s 0us/step
    13/13 [==============================] - 2s 107ms/step - loss: 2.3052 - acc: 0.1050 - val_loss: 2.3003 - val_acc: 0.1150





    <keras.callbacks.History at 0x7ff608121af0>



## 共享层
函数式 API 的另一个很好的用途是使用shared layers的模型。共享层是在同一个模型中多次重用的层实例，它们会学习与层计算图中的多个路径相对应的特征。

共享层通常用于对来自相似空间（例如，两个具有相似词汇的不同文本）的输入进行编码。它们可以实现在这些不同的输入之间共享信息，以及在更少的数据上训练这种模型。如果在其中的一个输入中看到了一个给定单词，那么将有利于处理通过共享层的所有输入。

要在函数式 API 中共享层，请多次调用同一个层实例。例如，下面是一个在两个不同文本输入之间共享的 Embedding 层：


```python
# Embedding for 1000 unique words mapped to 128-dimensional vectors
shared_embedding = layers.Embedding(1000, 128)

# Variable-length sequence of integers
text_input_a = keras.Input(shape=(None,), dtype="int32")

# Variable-length sequence of integers
text_input_b = keras.Input(shape=(None,), dtype="int32")

# Reuse the same layer to encode both inputs
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)
```

## 提取和重用计算图中各层的节点
由于要处理的层计算图是静态数据结构，可以对其进行访问和检查。而这就是将函数式模型绘制为图像的方式。

这也意味着您可以访问中间层的激活函数（计算图中的“节点”）并在其他地方重用它们，这对于特征提取之类的操作十分有用。

让我们来看一个例子。下面是一个 VGG19 模型，其权重已在 ImageNet 上进行了预训练：


```python
vgg19 = tf.keras.applications.VGG19()
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5
    574710816/574710816 [==============================] - 28s 0us/step


下面是通过查询计算图数据结构获得的模型的中间激活：


```python
features_list = [layer.output for layer in vgg19.layers]
features_list
```




    [<KerasTensor: shape=(None, 224, 224, 3) dtype=float32 (created by layer 'input_9')>,
     <KerasTensor: shape=(None, 224, 224, 64) dtype=float32 (created by layer 'block1_conv1')>,
     <KerasTensor: shape=(None, 224, 224, 64) dtype=float32 (created by layer 'block1_conv2')>,
     <KerasTensor: shape=(None, 112, 112, 64) dtype=float32 (created by layer 'block1_pool')>,
     <KerasTensor: shape=(None, 112, 112, 128) dtype=float32 (created by layer 'block2_conv1')>,
     <KerasTensor: shape=(None, 112, 112, 128) dtype=float32 (created by layer 'block2_conv2')>,
     <KerasTensor: shape=(None, 56, 56, 128) dtype=float32 (created by layer 'block2_pool')>,
     <KerasTensor: shape=(None, 56, 56, 256) dtype=float32 (created by layer 'block3_conv1')>,
     <KerasTensor: shape=(None, 56, 56, 256) dtype=float32 (created by layer 'block3_conv2')>,
     <KerasTensor: shape=(None, 56, 56, 256) dtype=float32 (created by layer 'block3_conv3')>,
     <KerasTensor: shape=(None, 56, 56, 256) dtype=float32 (created by layer 'block3_conv4')>,
     <KerasTensor: shape=(None, 28, 28, 256) dtype=float32 (created by layer 'block3_pool')>,
     <KerasTensor: shape=(None, 28, 28, 512) dtype=float32 (created by layer 'block4_conv1')>,
     <KerasTensor: shape=(None, 28, 28, 512) dtype=float32 (created by layer 'block4_conv2')>,
     <KerasTensor: shape=(None, 28, 28, 512) dtype=float32 (created by layer 'block4_conv3')>,
     <KerasTensor: shape=(None, 28, 28, 512) dtype=float32 (created by layer 'block4_conv4')>,
     <KerasTensor: shape=(None, 14, 14, 512) dtype=float32 (created by layer 'block4_pool')>,
     <KerasTensor: shape=(None, 14, 14, 512) dtype=float32 (created by layer 'block5_conv1')>,
     <KerasTensor: shape=(None, 14, 14, 512) dtype=float32 (created by layer 'block5_conv2')>,
     <KerasTensor: shape=(None, 14, 14, 512) dtype=float32 (created by layer 'block5_conv3')>,
     <KerasTensor: shape=(None, 14, 14, 512) dtype=float32 (created by layer 'block5_conv4')>,
     <KerasTensor: shape=(None, 7, 7, 512) dtype=float32 (created by layer 'block5_pool')>,
     <KerasTensor: shape=(None, 25088) dtype=float32 (created by layer 'flatten')>,
     <KerasTensor: shape=(None, 4096) dtype=float32 (created by layer 'fc1')>,
     <KerasTensor: shape=(None, 4096) dtype=float32 (created by layer 'fc2')>,
     <KerasTensor: shape=(None, 1000) dtype=float32 (created by layer 'predictions')>]



使用以下特征来创建新的特征提取模型，该模型会返回中间层激活的值：


```python
feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)

img = np.random.random((1, 224, 224, 3)).astype("float32")
extracted_features = feat_extraction_model(img)
extracted_features
```




    [<tf.Tensor: shape=(1, 224, 224, 3), dtype=float32, numpy=
     array([[[[0.13017905, 0.3140408 , 0.4452513 ],
              [0.27118048, 0.05233892, 0.28433752],
              [0.151976  , 0.4319893 , 0.55831224],
              ...,
              [0.8763184 , 0.59210414, 0.48051208],
              [0.6382841 , 0.27595252, 0.335217  ],
              [0.9471245 , 0.19854414, 0.38608423]],
     
             [[0.33069226, 0.8775885 , 0.6221657 ],
              [0.7106222 , 0.78345174, 0.00749704],
              [0.9664353 , 0.75632215, 0.8306189 ],
              ...,
              [0.16262703, 0.02946582, 0.24245262],
              [0.19082658, 0.03512435, 0.0944546 ],
              [0.43021145, 0.16367343, 0.10053137]],
     
             [[0.14758448, 0.3785106 , 0.08081373],
              [0.7142697 , 0.96772885, 0.72520125],
              [0.7151739 , 0.5310668 , 0.28563008],
              ...,
              [0.86766535, 0.21213572, 0.39406663],
              [0.25858757, 0.4508095 , 0.10999423],
              [0.23276809, 0.91577977, 0.8505227 ]],
     
             ...,
     
             [[0.896477  , 0.73856705, 0.7649225 ],
              [0.9518494 , 0.00515453, 0.8175194 ],
              [0.5856802 , 0.29118675, 0.74940467],
              ...,
              [0.5495355 , 0.22299871, 0.5635752 ],
              [0.1514617 , 0.9794774 , 0.50524765],
              [0.47208843, 0.37211388, 0.74204403]],
     
             [[0.92505926, 0.8738863 , 0.93863124],
              [0.01170055, 0.3014974 , 0.6899851 ],
              [0.38540393, 0.07260269, 0.4783115 ],
              ...,
              [0.76116997, 0.7427307 , 0.6039166 ],
              [0.68667305, 0.75901306, 0.3436342 ],
              [0.25761423, 0.5377181 , 0.90639645]],
     
             [[0.05283256, 0.7319009 , 0.4632635 ],
              [0.00825454, 0.8438331 , 0.39132172],
              [0.82150507, 0.8193277 , 0.6242451 ],
              ...,
              [0.56367123, 0.45427123, 0.9145866 ],
              [0.2997577 , 0.5574241 , 0.83395135],
              [0.08432637, 0.51181734, 0.79994243]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 224, 224, 64), dtype=float32, numpy=
     array([[[[0.        , 0.16935739, 0.11620638, ..., 0.672238  ,
               0.7104093 , 0.29829362],
              [0.        , 0.16610956, 0.10794577, ..., 0.23157293,
               0.9379545 , 0.26830715],
              [0.        , 0.27374652, 0.15361807, ..., 0.22884893,
               1.1647177 , 0.58144474],
              ...,
              [0.6089271 , 0.30468395, 0.51589483, ..., 0.52785164,
               1.5087794 , 1.5976946 ],
              [0.48940825, 0.35860822, 0.35617408, ..., 0.5151346 ,
               1.1203039 , 1.1937251 ],
              [0.98745894, 0.38476098, 0.43949002, ..., 0.81802785,
               1.4356737 , 1.5349693 ]],
     
             [[0.        , 0.19469267, 0.12806454, ..., 0.37192857,
               0.01608503, 0.        ],
              [0.        , 0.35744628, 0.40860206, ..., 0.        ,
               0.688666  , 0.588063  ],
              [0.26984823, 0.46497503, 0.46855572, ..., 0.        ,
               1.0739969 , 1.103521  ],
              ...,
              [1.5150251 , 0.34834158, 0.28901646, ..., 0.0344559 ,
               0.52895325, 0.59091544],
              [0.5970316 , 0.30652636, 0.        , ..., 0.        ,
               0.        , 0.        ],
              [1.2628582 , 0.38420585, 0.34443894, ..., 0.4728812 ,
               0.9749279 , 0.9252063 ]],
     
             [[0.14255059, 0.18296039, 0.        , ..., 0.29107416,
               0.        , 0.        ],
              [0.99075925, 0.49276248, 0.4042299 , ..., 0.        ,
               0.        , 0.14757422],
              [1.2974278 , 0.48886693, 0.48161212, ..., 0.        ,
               0.3710572 , 0.6795722 ],
              ...,
              [0.27612776, 0.42414498, 0.4563457 , ..., 0.06272173,
               1.2765841 , 0.99616534],
              [0.        , 0.3170625 , 0.29728463, ..., 0.04220045,
               0.86447245, 0.49523118],
              [1.0194026 , 0.46604678, 0.6845736 , ..., 0.55943644,
               1.7210656 , 1.7318432 ]],
     
             ...,
     
             [[0.        , 0.26779076, 0.14662553, ..., 0.17515653,
               0.53072834, 0.5122605 ],
              [0.8918414 , 0.18208691, 0.27691787, ..., 0.        ,
               1.0568904 , 1.2281761 ],
              [0.834016  , 0.16882578, 0.14814325, ..., 0.        ,
               0.13572145, 0.4638264 ],
              ...,
              [0.18492907, 0.2082398 , 0.20635922, ..., 0.        ,
               0.4998541 , 0.3963198 ],
              [0.19226855, 0.2350784 , 0.29324165, ..., 0.        ,
               0.81850064, 0.7210916 ],
              [1.6330204 , 0.25478572, 0.6203049 , ..., 0.32566983,
               1.861883  , 1.9923842 ]],
     
             [[0.5402831 , 0.19601098, 0.22371395, ..., 0.17417663,
               0.        , 0.35742512],
              [1.380472  , 0.17342941, 0.2847498 , ..., 0.        ,
               0.6099012 , 0.79369533],
              [0.5367186 , 0.2071499 , 0.10677229, ..., 0.        ,
               0.24384595, 0.16524923],
              ...,
              [0.2351363 , 0.22092283, 0.3364149 , ..., 0.        ,
               0.55712295, 0.5651017 ],
              [0.7486406 , 0.14624467, 0.38410378, ..., 0.        ,
               0.7706188 , 0.94969815],
              [2.1524565 , 0.21295106, 0.6882924 , ..., 0.26994002,
               1.7344701 , 2.0684447 ]],
     
             [[1.3937917 , 0.06742893, 0.17742942, ..., 0.5109157 ,
               0.        , 0.19581906],
              [1.8881254 , 0.07569986, 0.3413258 , ..., 0.21790487,
               0.08124119, 0.72920376],
              [1.5222595 , 0.23271263, 0.5257103 , ..., 0.4520628 ,
               0.53383803, 1.1454848 ],
              ...,
              [2.0142634 , 0.17388351, 0.42014375, ..., 0.15752989,
               0.07537782, 0.9142468 ],
              [2.3666656 , 0.10156836, 0.38995725, ..., 0.12344795,
               0.15963548, 1.0275183 ],
              [2.709778  , 0.13470155, 0.52846944, ..., 0.555496  ,
               0.9833439 , 1.6694252 ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 224, 224, 64), dtype=float32, numpy=
     array([[[[4.5180745 , 1.0949275 , 0.79241127, ..., 0.6357715 ,
               2.7536173 , 2.1064377 ],
              [3.512313  , 1.896004  , 1.9926039 , ..., 1.3994181 ,
               0.59110546, 2.5060642 ],
              [2.0786598 , 1.677726  , 2.8381388 , ..., 1.7165308 ,
               0.0468493 , 2.163971  ],
              ...,
              [0.        , 0.        , 2.7071629 , ..., 0.85360235,
               0.        , 1.5482597 ],
              [0.        , 0.        , 3.474638  , ..., 0.6841925 ,
               1.6399078 , 1.6609602 ],
              [0.        , 0.        , 2.588796  , ..., 0.65913093,
               0.        , 0.6154031 ]],
     
             [[2.6426136 , 0.23683846, 1.9278394 , ..., 1.2599351 ,
               5.4551287 , 0.70537835],
              [0.        , 0.13274843, 4.23329   , ..., 2.133433  ,
               6.3085947 , 0.15914673],
              [0.        , 0.        , 5.304811  , ..., 2.3156898 ,
               3.7272341 , 0.        ],
              ...,
              [0.42960805, 0.        , 4.3016214 , ..., 1.201988  ,
               0.        , 1.4214046 ],
              [3.6099954 , 0.14814979, 4.775835  , ..., 1.2791084 ,
               1.6535275 , 1.5975952 ],
              [0.7906373 , 1.1202734 , 3.3104122 , ..., 1.3105181 ,
               1.269518  , 0.34861046]],
     
             [[0.67295915, 0.        , 2.0503478 , ..., 0.497378  ,
               1.148889  , 0.64905864],
              [0.        , 0.        , 4.272726  , ..., 0.7571436 ,
               6.153383  , 0.42672247],
              [0.        , 0.        , 4.7255874 , ..., 0.77959394,
               4.5402017 , 0.5119392 ],
              ...,
              [5.480363  , 2.3661056 , 4.5351205 , ..., 2.0527267 ,
               1.0752435 , 0.49688727],
              [1.9451604 , 1.9970858 , 4.3641353 , ..., 1.826163  ,
               0.23988219, 0.5108921 ],
              [0.        , 1.1746922 , 2.6721308 , ..., 1.3521473 ,
               0.        , 0.        ]],
     
             ...,
     
             [[0.19350475, 0.20529038, 3.120259  , ..., 2.1299825 ,
               3.491582  , 0.5051839 ],
              [0.        , 0.        , 5.0837407 , ..., 3.1347156 ,
               0.        , 0.48587245],
              [0.        , 0.06039381, 5.042338  , ..., 2.3935976 ,
               0.        , 0.56989616],
              ...,
              [3.7074046 , 2.6023302 , 2.7746816 , ..., 3.088009  ,
               1.5121285 , 0.66381115],
              [0.        , 0.04508263, 2.4419162 , ..., 2.4377728 ,
               2.585263  , 0.71898776],
              [0.        , 0.        , 0.8245334 , ..., 2.081239  ,
               0.        , 0.        ]],
     
             [[0.        , 0.        , 1.7316284 , ..., 1.36082   ,
               4.8062835 , 0.9465789 ],
              [0.        , 0.        , 3.1091022 , ..., 2.46749   ,
               3.2971628 , 1.2743194 ],
              [2.0274425 , 0.29548568, 3.6223407 , ..., 2.4769335 ,
               0.31177908, 1.2767715 ],
              ...,
              [0.        , 0.        , 2.2346206 , ..., 2.66533   ,
               2.9760811 , 1.7828774 ],
              [0.        , 0.        , 2.0443957 , ..., 1.969854  ,
               4.913606  , 1.8608136 ],
              [0.        , 0.        , 0.6776755 , ..., 1.789721  ,
               1.9179853 , 0.47320914]],
     
             [[0.        , 0.        , 0.5117074 , ..., 0.5779263 ,
               0.389318  , 0.        ],
              [1.2305439 , 0.31086665, 1.2306137 , ..., 1.5328104 ,
               3.842388  , 0.        ],
              [1.0721917 , 1.4025929 , 1.9812275 , ..., 1.574572  ,
               2.61278   , 0.        ],
              ...,
              [0.        , 0.        , 0.86616874, ..., 1.0016699 ,
               2.444527  , 0.        ],
              [0.        , 0.        , 0.7621436 , ..., 0.9584446 ,
               4.503016  , 0.        ],
              [2.409283  , 0.06469011, 0.10378025, ..., 1.4486169 ,
               4.082558  , 0.        ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 112, 112, 64), dtype=float32, numpy=
     array([[[[4.5180745 , 1.896004  , 4.23329   , ..., 2.133433  ,
               6.3085947 , 2.5060642 ],
              [2.0786598 , 1.677726  , 6.0519576 , ..., 2.4396951 ,
               3.7272341 , 2.163971  ],
              [0.7431565 , 0.5944899 , 6.5484476 , ..., 2.25559   ,
               2.1600797 , 1.6227543 ],
              ...,
              [2.6336246 , 1.1559794 , 3.2029147 , ..., 2.329263  ,
               7.8923426 , 1.6952171 ],
              [0.42960805, 0.        , 4.3016214 , ..., 1.201988  ,
               1.2760328 , 1.5482597 ],
              [3.6099954 , 1.1202734 , 4.775835  , ..., 1.3105181 ,
               1.6535275 , 1.6609602 ]],
     
             [[2.5084934 , 2.346693  , 4.272726  , ..., 1.0502964 ,
               6.153383  , 0.6952521 ],
              [1.4694595 , 0.94521046, 4.8568087 , ..., 1.3249613 ,
               4.5402017 , 0.7744941 ],
              [3.6810305 , 1.4790969 , 5.350089  , ..., 1.7076646 ,
               3.457679  , 0.89041275],
              ...,
              [2.0139706 , 1.6940167 , 6.354194  , ..., 1.5861596 ,
               7.340322  , 0.8821302 ],
              [5.480363  , 2.3661056 , 6.2249794 , ..., 2.0527267 ,
               5.4618287 , 0.7163282 ],
              [1.9451604 , 1.9970858 , 4.3641353 , ..., 1.826163  ,
               3.5527627 , 0.530096  ]],
     
             [[2.523106  , 2.7160995 , 3.3058634 , ..., 2.2060637 ,
               3.8483956 , 0.7930502 ],
              [1.2174423 , 2.004176  , 3.6063232 , ..., 1.8963251 ,
               2.8370187 , 1.2335343 ],
              [3.4464607 , 2.099553  , 4.137314  , ..., 1.5358232 ,
               3.3105779 , 0.95990413],
              ...,
              [2.6839569 , 2.2497735 , 6.4294567 , ..., 2.1336033 ,
               0.        , 1.1212852 ],
              [1.3000307 , 2.1195679 , 6.7693534 , ..., 0.82797664,
               1.8825749 , 1.0688224 ],
              [2.5250492 , 0.9266001 , 4.8037796 , ..., 1.5832295 ,
               5.241335  , 1.0907803 ]],
     
             ...,
     
             [[3.761067  , 3.5570958 , 5.8009    , ..., 3.2092216 ,
               0.8611912 , 0.93448216],
              [1.0529904 , 2.8179512 , 4.815053  , ..., 2.9798596 ,
               2.895484  , 0.68751544],
              [2.101317  , 0.7310456 , 4.8919535 , ..., 1.8592391 ,
               1.8692927 , 0.9209482 ],
              ...,
              [1.1340795 , 1.6589272 , 3.1271489 , ..., 2.8368971 ,
               1.6360453 , 0.9392844 ],
              [3.2118993 , 2.256118  , 5.2372007 , ..., 3.03717   ,
               3.6299305 , 1.1054316 ],
              [0.4970618 , 0.17223841, 4.464454  , ..., 2.7121649 ,
               7.00286   , 0.947967  ]],
     
             [[2.6886783 , 1.783992  , 5.4680266 , ..., 3.3895025 ,
               3.491582  , 0.98954314],
              [0.        , 0.6701575 , 5.198848  , ..., 2.6246428 ,
               3.2165012 , 0.8428851 ],
              [2.7522411 , 1.2862656 , 4.9612474 , ..., 2.2356825 ,
               1.6827106 , 1.139195  ],
              ...,
              [0.43601388, 0.        , 2.630549  , ..., 2.950732  ,
               2.7194831 , 0.906645  ],
              [5.090994  , 3.2332668 , 4.1167116 , ..., 3.4714568 ,
               2.0805545 , 0.9381408 ],
              [0.10271132, 1.40958   , 3.578025  , ..., 2.8804133 ,
               2.585263  , 0.92837614]],
     
             [[1.2305439 , 0.31086665, 3.1091022 , ..., 2.46749   ,
               4.8062835 , 1.2743194 ],
              [2.0274425 , 1.4025929 , 4.1657147 , ..., 2.4769335 ,
               2.61278   , 1.2767715 ],
              [3.9052196 , 1.4675181 , 4.8802686 , ..., 2.742715  ,
               2.9351954 , 1.2952733 ],
              ...,
              [3.5159397 , 0.6049152 , 1.7314365 , ..., 2.722821  ,
               4.227475  , 1.858139  ],
              [2.2041385 , 1.0126319 , 2.2346206 , ..., 2.8851557 ,
               2.9760811 , 1.7828774 ],
              [2.409283  , 0.06469011, 2.0443957 , ..., 1.969854  ,
               4.913606  , 1.8608136 ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 112, 112, 128), dtype=float32, numpy=
     array([[[[ 0.        ,  0.        ,  0.        , ...,  4.5862613 ,
                0.        , 15.152903  ],
              [ 0.        ,  0.        ,  2.5372415 , ...,  4.4138255 ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  3.4914105 , ...,  6.758958  ,
                0.        ,  0.6463379 ],
              ...,
              [ 0.        ,  0.        ,  1.3635056 , ...,  8.724387  ,
                0.        ,  3.2686465 ],
              [ 0.        ,  0.        ,  6.932256  , ...,  6.1326513 ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  6.8891478 , ...,  7.6662254 ,
                0.        ,  0.        ]],
     
             [[ 0.        ,  4.6926546 ,  0.        , ...,  4.731545  ,
                1.9322982 , 17.778883  ],
              [ 0.        ,  4.596176  ,  0.8890343 , ...,  3.5051365 ,
                2.2887902 ,  0.        ],
              [ 0.        ,  2.4115262 ,  1.9119732 , ...,  4.7378764 ,
                0.        ,  0.8526042 ],
              ...,
              [ 0.        ,  2.1749907 ,  0.        , ...,  5.0113745 ,
                6.1620784 ,  0.78528565],
              [ 0.        ,  3.4160163 ,  2.8432899 , ...,  5.506218  ,
                0.        ,  0.33189005],
              [ 0.        ,  0.43244585,  8.939903  , ..., 13.126759  ,
                2.2978566 ,  0.        ]],
     
             [[ 0.        ,  0.13278636,  0.        , ...,  7.4448953 ,
                0.        , 15.222713  ],
              [ 0.        ,  0.        ,  0.        , ...,  2.257878  ,
                0.        ,  0.43160647],
              [ 0.        ,  0.        ,  0.48992202, ...,  6.3459854 ,
                0.8869931 ,  0.        ],
              ...,
              [ 0.        ,  0.        ,  1.6883659 , ...,  3.0293949 ,
                0.        ,  0.53719276],
              [ 0.        ,  0.        ,  0.        , ...,  6.084799  ,
                2.3960884 ,  0.        ],
              [ 0.        ,  1.3952832 ,  6.257917  , ..., 10.278513  ,
                5.9428864 ,  0.        ]],
     
             ...,
     
             [[ 0.        ,  1.2069042 ,  0.        , ...,  4.2466326 ,
                0.        , 15.529216  ],
              [ 0.        ,  2.1575298 ,  0.        , ...,  6.371885  ,
                5.44426   ,  0.3080222 ],
              [ 0.        ,  1.827267  ,  1.3734864 , ...,  4.114907  ,
                0.        ,  1.2563378 ],
              ...,
              [ 0.        ,  0.        ,  0.33497864, ...,  6.3972454 ,
                0.        ,  1.5263575 ],
              [ 0.        ,  0.        ,  0.        , ...,  5.611367  ,
                1.576901  ,  1.5306321 ],
              [ 0.        ,  0.5057845 ,  5.5105653 , ..., 13.5244665 ,
                5.9099646 ,  0.        ]],
     
             [[ 0.        ,  0.785972  ,  0.        , ...,  6.3257704 ,
                0.        , 15.474985  ],
              [ 0.        ,  1.038249  ,  0.        , ...,  5.6915545 ,
                5.185066  ,  0.        ],
              [ 0.        ,  1.2681725 ,  0.95697796, ...,  3.149223  ,
                0.        ,  1.5513989 ],
              ...,
              [ 0.        ,  0.57604337,  2.0686383 , ...,  9.769748  ,
                0.        ,  2.281017  ],
              [ 0.        ,  0.        ,  0.848279  , ...,  5.4564514 ,
                0.        ,  1.6535162 ],
              [ 0.        ,  0.        ,  5.4885583 , ..., 13.325162  ,
                4.732159  ,  0.        ]],
     
             [[ 0.        ,  3.7255185 ,  0.        , ..., 10.049769  ,
                0.        , 10.882535  ],
              [ 0.        ,  3.3047879 ,  1.7381728 , ...,  3.7149057 ,
                0.3403681 ,  0.        ],
              [ 0.        ,  2.8765595 ,  0.20335421, ...,  5.6937017 ,
                0.        ,  2.0574467 ],
              ...,
              [ 0.        ,  5.1957684 ,  0.567664  , ...,  5.4367547 ,
                0.        ,  3.446925  ],
              [ 0.        ,  5.5735173 ,  0.49431893, ...,  9.020008  ,
                1.5832783 ,  2.8970397 ],
              [ 0.        ,  3.76688   ,  7.5356894 , ..., 13.705661  ,
                3.0032196 ,  0.        ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 112, 112, 128), dtype=float32, numpy=
     array([[[[ 0.        ,  3.4703054 ,  0.38254356, ...,  0.9157389 ,
                0.        ,  0.        ],
              [ 1.149838  ,  0.        ,  0.        , ...,  9.186115  ,
                0.        ,  0.        ],
              [ 1.952816  ,  0.        ,  0.        , ...,  8.65875   ,
                0.        ,  0.        ],
              ...,
              [ 9.3127775 ,  0.        ,  0.        , ...,  7.6015024 ,
                0.        ,  0.        ],
              [ 4.2847424 ,  0.54633945,  0.        , ...,  9.982564  ,
                0.        ,  1.592062  ],
              [ 2.5778081 ,  0.        ,  2.1202426 , ...,  9.574043  ,
                0.        ,  3.959626  ]],
     
             [[ 0.        ,  0.886944  ,  0.5060447 , ...,  3.9849215 ,
               15.940543  ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ..., 15.126825  ,
                4.044423  ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ..., 15.259501  ,
                4.902989  ,  0.        ],
              ...,
              [ 2.3539891 ,  0.        ,  0.        , ..., 15.865889  ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ..., 16.86783   ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.85912275, ..., 15.38466   ,
                0.        ,  4.678432  ]],
     
             [[ 0.        ,  2.206544  ,  1.2298328 , ...,  2.8059115 ,
               17.940395  ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ..., 14.556754  ,
                7.822493  ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ..., 14.200677  ,
                6.7132235 ,  0.        ],
              ...,
              [ 0.77146345,  0.        ,  0.        , ..., 14.996774  ,
                1.4994199 ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ..., 16.155668  ,
                0.        ,  0.        ],
              [ 0.89767367,  0.        ,  0.11935289, ..., 13.35041   ,
                0.        ,  3.6172318 ]],
     
             ...,
     
             [[ 0.        ,  0.5860233 ,  6.186368  , ...,  4.6802754 ,
               22.95292   ,  0.        ],
              [10.793243  ,  0.        ,  4.542647  , ..., 15.462351  ,
               12.715335  ,  0.        ],
              [10.353613  ,  0.        ,  0.        , ..., 14.296093  ,
                9.348029  ,  0.        ],
              ...,
              [ 4.5777984 ,  0.67466325,  0.        , ..., 24.643406  ,
                7.0524573 ,  0.        ],
              [ 9.626147  ,  0.        ,  0.        , ..., 21.462467  ,
                0.5035648 ,  0.        ],
              [13.665979  ,  0.        ,  0.33897537, ..., 13.649668  ,
                0.        ,  0.        ]],
     
             [[ 0.        ,  3.1009564 ,  9.98919   , ...,  3.625688  ,
               24.330732  ,  0.        ],
              [ 4.784755  ,  0.        ,  0.        , ..., 11.174282  ,
               15.551071  ,  0.        ],
              [ 4.9561377 ,  0.        ,  0.        , ...,  9.253525  ,
               13.842019  ,  0.        ],
              ...,
              [ 0.896016  ,  1.6873375 ,  0.        , ..., 16.987286  ,
                7.4703574 ,  0.        ],
              [ 3.7568696 ,  0.        ,  1.5667013 , ..., 16.736124  ,
                3.3658752 ,  0.        ],
              [ 8.233102  ,  0.        ,  2.0429373 , ..., 13.043998  ,
                0.        ,  0.        ]],
     
             [[ 0.        ,  1.1338186 ,  3.549194  , ...,  0.        ,
               19.495165  ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  3.2951064 ,
               13.958234  ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  2.2895875 ,
               13.934224  ,  0.        ],
              ...,
              [ 0.        ,  0.        ,  0.        , ...,  4.9069643 ,
                9.251039  ,  0.        ],
              [ 0.        ,  0.        ,  5.7732964 , ...,  6.035894  ,
                5.7688007 ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  4.6819744 ,
                0.        ,  0.        ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 56, 56, 128), dtype=float32, numpy=
     array([[[[ 1.149838  ,  3.4703054 ,  0.5060447 , ..., 15.126825  ,
               15.940543  ,  0.        ],
              [ 5.1985273 ,  0.        ,  2.3238308 , ..., 16.79499   ,
                8.857392  ,  0.        ],
              [ 6.6173153 ,  1.9958253 ,  1.8151399 , ..., 15.722354  ,
                1.5725658 ,  1.6890635 ],
              ...,
              [ 4.0063    ,  5.8332562 ,  0.8405513 , ..., 16.674232  ,
                1.4828691 ,  0.        ],
              [ 9.3127775 ,  0.        ,  0.        , ..., 16.730787  ,
                6.532488  ,  0.        ],
              [ 4.2847424 ,  0.54633945,  2.1202426 , ..., 16.86783   ,
                0.        ,  4.678432  ]],
     
             [[ 4.4902544 ,  5.739122  ,  5.16446   , ..., 14.556754  ,
               17.940395  ,  0.        ],
              [ 0.        ,  0.        ,  2.4367428 , ..., 14.200677  ,
               11.473945  ,  0.        ],
              [ 0.        ,  2.2101321 ,  1.9572705 , ..., 13.60562   ,
               10.311266  ,  0.        ],
              ...,
              [ 0.        ,  0.        ,  1.0687582 , ..., 15.768269  ,
               11.046063  ,  0.        ],
              [ 5.312985  ,  0.        ,  0.        , ..., 14.996774  ,
                5.149448  ,  0.        ],
              [ 3.4372187 ,  0.        ,  0.8106098 , ..., 16.155668  ,
                0.        ,  3.7310922 ]],
     
             [[11.589242  ,  9.656948  ,  6.401707  , ..., 13.436615  ,
               19.292202  ,  0.        ],
              [ 8.324088  ,  0.        ,  4.068875  , ..., 15.485772  ,
                8.086813  ,  0.        ],
              [ 0.        ,  1.9512844 ,  4.529129  , ..., 15.251849  ,
               11.168576  ,  0.        ],
              ...,
              [ 6.875134  ,  0.        ,  0.        , ..., 16.092325  ,
                6.6539145 ,  0.        ],
              [11.961515  ,  0.        ,  3.670983  , ..., 15.716295  ,
                6.7577896 ,  0.        ],
              [10.476205  ,  0.        ,  1.6424192 , ..., 14.28093   ,
                0.        ,  5.5038204 ]],
     
             ...,
     
             [[ 9.232451  ,  3.8480592 ,  0.6914541 , ..., 17.110373  ,
               19.11056   ,  0.        ],
              [ 2.704014  ,  0.        ,  5.3499722 , ..., 14.317195  ,
               11.327838  ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ..., 13.789909  ,
               10.826119  ,  0.        ],
              ...,
              [ 0.2820368 ,  0.        ,  1.8038633 , ..., 20.623573  ,
                9.358798  ,  0.        ],
              [ 3.4345024 ,  0.        ,  0.        , ..., 20.71817   ,
                8.362825  ,  0.        ],
              [ 6.610878  ,  0.        ,  1.7630774 , ..., 20.448017  ,
                0.44170028,  0.        ]],
     
             [[11.722787  ,  0.5860233 ,  7.4202113 , ..., 17.289122  ,
               22.95292   ,  0.        ],
              [10.353613  ,  0.        ,  0.        , ..., 16.321295  ,
               11.178618  ,  0.        ],
              [ 5.769253  ,  0.        ,  6.547338  , ..., 16.259142  ,
               12.31994   ,  0.        ],
              ...,
              [11.762082  ,  0.        ,  5.050907  , ..., 19.226646  ,
                8.718595  ,  0.        ],
              [ 6.898691  ,  0.67466325,  0.        , ..., 25.253477  ,
                7.3780303 ,  0.        ],
              [13.665979  ,  0.        ,  0.33897537, ..., 22.714443  ,
                0.5035648 ,  0.        ]],
     
             [[ 4.784755  ,  3.1009564 ,  9.98919   , ..., 11.174282  ,
               24.330732  ,  0.        ],
              [ 6.922037  ,  0.        ,  7.4201627 , ..., 11.688176  ,
               16.748352  ,  0.        ],
              [ 8.899505  ,  0.        ,  0.        , ..., 13.6037655 ,
               14.035448  ,  0.        ],
              ...,
              [ 1.7377161 ,  0.        ,  4.771134  , ..., 16.1094    ,
               14.170901  ,  0.        ],
              [ 3.0455284 ,  1.6873375 ,  0.        , ..., 18.226152  ,
               10.395574  ,  0.        ],
              [ 8.233102  ,  0.        ,  5.7732964 , ..., 16.736124  ,
                5.7688007 ,  0.        ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 56, 56, 256), dtype=float32, numpy=
     array([[[[0.00000000e+00, 5.44541702e-02, 1.95365831e-01, ...,
               1.90980613e-01, 6.51013279e+00, 1.61022873e+01],
              [0.00000000e+00, 0.00000000e+00, 9.09663677e-01, ...,
               5.34555353e-02, 4.32918072e+00, 1.52587919e+01],
              [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ...,
               5.38616478e-01, 4.29967642e+00, 1.64190960e+01],
              ...,
              [0.00000000e+00, 3.72104192e+00, 0.00000000e+00, ...,
               0.00000000e+00, 3.77227783e+00, 1.63943844e+01],
              [2.54344583e-01, 1.03367348e+01, 1.48700297e+00, ...,
               0.00000000e+00, 3.32707834e+00, 1.66397285e+01],
              [0.00000000e+00, 9.07818031e+00, 0.00000000e+00, ...,
               1.63358569e+00, 0.00000000e+00, 1.64685898e+01]],
     
             [[0.00000000e+00, 0.00000000e+00, 4.73896980e+00, ...,
               2.67153311e+00, 4.17917299e+00, 1.13874235e+01],
              [0.00000000e+00, 0.00000000e+00, 6.05313492e+00, ...,
               2.25018024e+00, 1.73292482e+00, 6.30916977e+00],
              [2.33958983e+00, 0.00000000e+00, 5.86809778e+00, ...,
               3.21767545e+00, 7.89680719e+00, 7.81490993e+00],
              ...,
              [3.17469335e+00, 3.77861977e+00, 2.89126134e+00, ...,
               0.00000000e+00, 2.42918178e-01, 1.04114141e+01],
              [8.62679672e+00, 1.30267601e+01, 7.79100704e+00, ...,
               0.00000000e+00, 1.58988512e+00, 9.78570461e+00],
              [9.12915245e-02, 1.03002853e+01, 6.35660028e+00, ...,
               4.18544674e+00, 0.00000000e+00, 1.28217735e+01]],
     
             [[1.21790767e+00, 0.00000000e+00, 1.62147450e+00, ...,
               3.08772516e+00, 5.50587416e+00, 8.42886448e+00],
              [0.00000000e+00, 0.00000000e+00, 3.12082797e-01, ...,
               1.55376983e+00, 0.00000000e+00, 2.42636099e-01],
              [1.75181496e+00, 0.00000000e+00, 0.00000000e+00, ...,
               2.03927374e+00, 7.01137424e-01, 2.08962178e+00],
              ...,
              [1.89118505e-01, 2.65709925e+00, 0.00000000e+00, ...,
               0.00000000e+00, 1.01434037e-01, 7.04449463e+00],
              [7.14168453e+00, 7.03944254e+00, 0.00000000e+00, ...,
               0.00000000e+00, 4.37591267e+00, 5.89259052e+00],
              [0.00000000e+00, 5.85511398e+00, 1.42455113e+00, ...,
               1.96501076e+00, 9.38897192e-01, 1.29041681e+01]],
     
             ...,
     
             [[1.84192467e+00, 0.00000000e+00, 3.93919492e+00, ...,
               4.72802544e+00, 5.79225111e+00, 1.04630222e+01],
              [1.33489311e-01, 0.00000000e+00, 2.35792112e+00, ...,
               5.96302795e+00, 0.00000000e+00, 5.48860836e+00],
              [1.70416725e+00, 0.00000000e+00, 0.00000000e+00, ...,
               2.97141409e+00, 6.11433458e+00, 7.71024132e+00],
              ...,
              [2.38607264e+00, 0.00000000e+00, 0.00000000e+00, ...,
               5.45672059e-01, 1.13251839e+01, 9.62379646e+00],
              [8.31085205e+00, 1.03194058e+00, 8.77592623e-01, ...,
               0.00000000e+00, 7.48806620e+00, 1.00676403e+01],
              [0.00000000e+00, 4.70677423e+00, 9.40856755e-01, ...,
               1.38733292e+00, 2.63144064e+00, 1.66908913e+01]],
     
             [[3.39013624e+00, 0.00000000e+00, 3.49235630e+00, ...,
               3.11648035e+00, 9.10262394e+00, 1.01162491e+01],
              [2.10002565e+00, 0.00000000e+00, 3.71504331e+00, ...,
               4.33523560e+00, 0.00000000e+00, 2.82330227e+00],
              [3.03692055e+00, 0.00000000e+00, 1.07359850e+00, ...,
               2.07586575e+00, 0.00000000e+00, 6.03477669e+00],
              ...,
              [4.25981474e+00, 0.00000000e+00, 2.13968372e+00, ...,
               2.48494864e+00, 5.56638098e+00, 8.62255096e+00],
              [9.04267216e+00, 6.12785280e-01, 3.52117181e+00, ...,
               4.86989379e-01, 4.82749033e+00, 1.03312340e+01],
              [0.00000000e+00, 3.25201607e+00, 2.40251017e+00, ...,
               2.34774604e-01, 4.71830463e+00, 1.74078865e+01]],
     
             [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ...,
               3.63562524e-01, 1.54434347e+01, 5.26965809e+00],
              [1.81425512e+00, 0.00000000e+00, 0.00000000e+00, ...,
               1.05222249e+00, 8.70715237e+00, 0.00000000e+00],
              [1.54847145e+00, 0.00000000e+00, 0.00000000e+00, ...,
               0.00000000e+00, 3.19010925e+00, 1.40385479e-02],
              ...,
              [1.84384596e+00, 0.00000000e+00, 0.00000000e+00, ...,
               1.15630662e+00, 1.09746628e+01, 3.01124072e+00],
              [4.08529568e+00, 2.61603206e-01, 0.00000000e+00, ...,
               0.00000000e+00, 1.16035480e+01, 1.48338437e+00],
              [0.00000000e+00, 5.56451142e-01, 0.00000000e+00, ...,
               0.00000000e+00, 1.08813858e+01, 8.96765232e+00]]]],
           dtype=float32)>,
     <tf.Tensor: shape=(1, 56, 56, 256), dtype=float32, numpy=
     array([[[[ 5.1608844 ,  0.        ,  5.519399  , ...,  0.        ,
               11.030516  ,  3.08187   ],
              [ 0.        ,  0.        ,  1.2304091 , ...,  0.        ,
               10.817544  ,  6.095614  ],
              [ 9.192054  ,  0.        ,  2.8806245 , ...,  0.        ,
                7.6896377 ,  4.0027003 ],
              ...,
              [ 6.49898   ,  0.        ,  2.6407619 , ...,  0.        ,
                5.819504  ,  2.6000385 ],
              [ 2.2630048 ,  0.        ,  3.8281956 , ...,  0.        ,
                7.7866826 ,  5.0391226 ],
              [ 2.3696704 ,  0.18827394,  6.706594  , ...,  0.        ,
                1.7440842 ,  3.961669  ]],
     
             [[ 3.5744538 ,  3.8718243 ,  2.8466012 , ...,  0.        ,
               12.236501  ,  0.9054576 ],
              [ 0.        ,  1.7511569 ,  0.        , ...,  0.        ,
               13.043017  ,  4.769533  ],
              [ 3.239162  ,  1.2282867 ,  0.        , ...,  0.        ,
                8.041148  ,  0.        ],
              ...,
              [ 0.        ,  9.812641  ,  0.        , ...,  0.        ,
                3.3169043 ,  3.217558  ],
              [ 0.        ,  6.267332  ,  0.        , ...,  0.        ,
                8.643767  ,  3.3981094 ],
              [ 0.        ,  6.12286   ,  8.387619  , ...,  0.        ,
                1.4342731 ,  0.06358207]],
     
             [[ 9.20747   ,  7.2810035 ,  4.25938   , ...,  1.3619596 ,
                7.7547846 ,  0.        ],
              [ 0.        ,  1.1021837 ,  0.        , ...,  1.0860376 ,
                9.026843  ,  2.5996122 ],
              [ 0.        ,  0.        ,  0.        , ...,  3.1165566 ,
                1.0493836 ,  0.        ],
              ...,
              [ 0.        ,  3.3367176 ,  0.        , ...,  0.54501784,
                3.6724026 ,  0.15478742],
              [ 0.        ,  2.0897121 ,  0.        , ...,  0.        ,
                8.777891  ,  0.        ],
              [ 0.        ,  5.7255006 ,  8.512261  , ...,  0.        ,
                1.944203  ,  0.        ]],
     
             ...,
     
             [[ 8.316358  ,  8.611217  ,  3.8402667 , ...,  0.67139673,
                9.803563  ,  0.        ],
              [ 0.        ,  7.4210987 ,  0.        , ...,  0.49435884,
                5.3189216 ,  2.1595197 ],
              [ 0.        ,  3.993085  ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              ...,
              [ 0.        ,  7.192813  ,  0.        , ...,  0.        ,
                2.8999507 ,  0.        ],
              [ 0.        , 12.097452  ,  0.        , ...,  0.        ,
                3.2388544 ,  0.        ],
              [ 5.725947  , 13.224792  , 11.609097  , ...,  0.        ,
                1.0727384 ,  0.        ]],
     
             [[13.035667  , 14.228443  ,  1.6702209 , ...,  3.5865834 ,
               12.401319  ,  0.        ],
              [ 0.        , 12.872138  ,  0.        , ...,  3.4025092 ,
                9.918149  ,  2.8407311 ],
              [ 0.        ,  8.359969  ,  0.        , ...,  1.5626814 ,
                2.6215155 ,  0.        ],
              ...,
              [ 0.        , 11.664688  ,  0.        , ...,  2.0427587 ,
                7.1850286 ,  0.5095782 ],
              [ 0.        , 14.595126  ,  0.        , ...,  1.9677726 ,
                6.8485475 ,  0.        ],
              [ 6.136234  , 16.83782   ,  8.313295  , ...,  0.        ,
                0.        ,  0.        ]],
     
             [[21.862137  , 13.306377  ,  1.6021483 , ...,  6.6890926 ,
               10.8963    ,  0.        ],
              [ 7.082569  , 13.421529  ,  0.6950921 , ...,  5.903715  ,
               10.591244  ,  2.7621138 ],
              [ 0.        , 10.197477  ,  1.8024551 , ...,  4.1789412 ,
                7.48565   ,  1.8678828 ],
              ...,
              [15.660954  , 10.482295  ,  0.89615536, ...,  1.6202464 ,
               12.31611   ,  2.9451542 ],
              [15.232232  ,  9.921546  ,  0.        , ...,  4.4992924 ,
                9.091634  ,  1.142001  ],
              [15.383014  , 12.937381  ,  5.433925  , ...,  4.8104815 ,
                0.07194307,  0.        ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 56, 56, 256), dtype=float32, numpy=
     array([[[[11.787364  ,  6.8622003 ,  9.042944  , ..., 10.176204  ,
               12.138853  ,  4.0985694 ],
              [11.692859  ,  7.4659376 , 11.637601  , ..., 14.200542  ,
               11.492658  ,  4.5584564 ],
              [13.742243  ,  8.121534  , 10.243381  , ..., 13.978197  ,
                4.2728734 ,  2.5632033 ],
              ...,
              [21.365978  , 17.536955  , 10.712654  , ...,  8.460585  ,
                9.231945  ,  0.        ],
              [29.847382  , 21.86624   ,  8.138429  , ...,  8.947056  ,
               11.341406  ,  1.6761035 ],
              [21.077213  , 15.306754  ,  3.1619272 , ...,  9.785701  ,
                9.706273  ,  0.80001974]],
     
             [[ 7.377816  , 11.502314  ,  9.247239  , ...,  8.675898  ,
               17.498062  ,  2.9441755 ],
              [ 4.710497  , 15.944716  ,  9.91812   , ...,  9.614695  ,
               16.229155  ,  2.5947425 ],
              [ 6.5066285 , 15.675425  ,  6.1113496 , ...,  6.498026  ,
                6.4689803 ,  0.        ],
              ...,
              [15.131392  , 22.677824  ,  5.4569197 , ...,  3.6639776 ,
               12.315567  ,  2.425912  ],
              [31.662308  , 29.00169   ,  5.661672  , ...,  6.260671  ,
               13.031429  ,  3.463537  ],
              [26.946064  , 20.8085    ,  3.3470418 , ..., 10.676118  ,
               11.710442  ,  0.        ]],
     
             [[ 0.9243362 , 13.14668   ,  7.2141895 , ...,  9.507789  ,
               14.299077  ,  0.        ],
              [ 0.        , 14.81383   ,  5.4211984 , ...,  9.489682  ,
               13.378039  ,  0.        ],
              [ 0.        , 12.118193  ,  0.6107643 , ...,  5.3094773 ,
                7.3522544 ,  0.        ],
              ...,
              [ 5.0838437 , 13.033725  ,  7.2109475 , ...,  4.85782   ,
                9.730728  ,  1.2799122 ],
              [11.640635  , 20.833065  ,  9.308368  , ...,  9.112449  ,
                6.569954  ,  3.3885207 ],
              [16.489834  , 14.017948  ,  7.5656147 , ..., 13.6262045 ,
                6.648414  ,  0.        ]],
     
             ...,
     
             [[ 0.59722686, 11.584068  ,  4.866275  , ..., 12.420334  ,
                9.34918   ,  0.        ],
              [ 0.        , 10.216013  ,  3.136321  , ..., 12.702072  ,
               10.738895  ,  0.        ],
              [ 0.        , 10.197306  ,  1.9788619 , ...,  7.3023953 ,
                7.2656283 ,  0.        ],
              ...,
              [ 0.        , 11.677664  ,  9.088487  , ..., 10.506215  ,
                1.9302859 ,  0.        ],
              [ 9.448083  , 11.769036  , 11.673853  , ...,  8.255848  ,
                2.4870358 ,  0.35101235],
              [16.308334  ,  8.105105  ,  8.061435  , ..., 12.504377  ,
                2.430953  ,  0.        ]],
     
             [[ 0.        , 14.87277   ,  7.9773464 , ..., 13.9077215 ,
                8.38645   ,  0.        ],
              [ 0.        , 15.370624  ,  7.2648034 , ..., 16.584965  ,
               13.190425  ,  0.        ],
              [ 0.        , 15.805453  ,  7.036796  , ..., 11.166724  ,
               16.136177  ,  0.        ],
              ...,
              [ 0.        , 11.4589    , 10.521035  , ..., 12.84533   ,
                7.6294847 ,  1.5253088 ],
              [ 1.9947873 , 12.214679  , 12.534344  , ...,  9.876199  ,
                8.110056  ,  6.3070335 ],
              [12.026678  , 11.463681  ,  8.927258  , ..., 11.537788  ,
                5.4108176 ,  0.        ]],
     
             [[ 5.4111805 , 11.082496  ,  7.2500515 , ..., 10.765428  ,
                5.2771297 ,  0.        ],
              [ 3.5402567 , 13.436274  ,  7.7999277 , ..., 13.993546  ,
               11.565496  ,  0.        ],
              [ 0.65319407, 14.420929  ,  7.384796  , ...,  9.98967   ,
               18.38823   ,  2.9804084 ],
              ...,
              [ 5.1185493 , 10.318114  , 15.767559  , ...,  9.253854  ,
               10.357399  ,  0.15248717],
              [ 3.2376125 , 14.209613  , 14.958178  , ...,  6.993283  ,
               10.634066  ,  2.989595  ],
              [ 7.997266  , 13.331049  , 10.200458  , ...,  7.536372  ,
                6.7792764 ,  0.        ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 56, 56, 256), dtype=float32, numpy=
     array([[[[ 7.3460374 , 21.373398  ,  2.25133   , ..., 10.61806   ,
                0.        ,  6.9715033 ],
              [13.905667  , 58.035545  ,  0.        , ...,  0.        ,
                0.        ,  4.195433  ],
              [16.462894  , 58.147404  ,  0.        , ...,  0.        ,
                0.        ,  4.738035  ],
              ...,
              [17.226122  , 55.223343  ,  1.6172324 , ...,  0.        ,
                0.        ,  3.6192179 ],
              [13.952495  , 66.4401    ,  4.707214  , ...,  0.        ,
                3.023843  ,  4.883298  ],
              [11.983194  , 84.87999   ,  4.523341  , ...,  0.        ,
               10.2836685 ,  0.25376615]],
     
             [[ 1.7516085 , 14.223699  , 15.849893  , ..., 30.157932  ,
                0.        ,  1.9055636 ],
              [ 3.4399729 , 46.48538   , 13.811924  , ...,  5.8981605 ,
                0.        ,  0.        ],
              [ 2.5468729 , 37.243187  , 11.471322  , ...,  0.        ,
                0.        ,  4.0596194 ],
              ...,
              [25.246674  , 26.734892  , 21.483719  , ...,  7.415467  ,
                0.        ,  1.7219633 ],
              [21.809414  , 42.33206   , 25.393993  , ...,  6.1818347 ,
                1.0533552 ,  3.525892  ],
              [15.814945  , 83.39274   , 19.22472   , ...,  1.3994433 ,
                3.3087487 ,  3.242938  ]],
     
             [[ 6.472588  , 18.422697  , 13.598929  , ..., 36.290955  ,
                0.        ,  3.0761397 ],
              [ 8.138961  , 48.221222  ,  6.25069   , ...,  4.03724   ,
                0.        ,  0.60855085],
              [ 1.4341643 , 33.45469   ,  1.3071035 , ...,  0.        ,
                0.        ,  5.8190064 ],
              ...,
              [35.046417  , 23.802664  ,  8.801834  , ...,  0.        ,
                0.        ,  9.017897  ],
              [34.44723   , 33.9309    , 15.3065195 , ...,  5.562992  ,
                0.        ,  9.161784  ],
              [22.291536  , 77.83233   , 14.218039  , ...,  5.400944  ,
                0.        ,  5.4291983 ]],
     
             ...,
     
             [[ 2.3117757 , 17.508362  ,  7.9303365 , ..., 27.404055  ,
                0.        ,  1.2637705 ],
              [ 8.220697  , 46.90642   ,  0.41310596, ...,  3.962065  ,
                0.        ,  0.        ],
              [10.083044  , 37.81973   ,  0.        , ...,  2.6641848 ,
                0.        ,  0.        ],
              ...,
              [32.99098   , 32.649834  ,  1.9000534 , ...,  5.2125497 ,
                0.        ,  0.66527736],
              [36.0115    , 36.178013  , 10.077438  , ..., 18.05717   ,
                0.        ,  0.        ],
              [24.125608  , 75.41097   , 10.817169  , ..., 23.293133  ,
                0.        ,  0.        ]],
     
             [[ 4.0730753 , 13.674155  ,  9.010488  , ..., 25.065252  ,
                2.6899312 ,  2.8166153 ],
              [13.79315   , 47.67468   ,  0.        , ...,  1.1463702 ,
                0.        ,  0.        ],
              [21.74248   , 47.830643  ,  0.        , ...,  0.        ,
                0.        ,  0.11377796],
              ...,
              [46.747837  , 29.815218  ,  0.        , ...,  2.267051  ,
                0.        ,  1.6578096 ],
              [51.25247   , 37.898808  ,  2.7273586 , ..., 12.5783615 ,
                0.        ,  0.        ],
              [33.55673   , 71.787285  ,  5.0050025 , ..., 21.104273  ,
                0.        ,  1.0477773 ]],
     
             [[ 0.        ,  0.30051973, 12.265128  , ..., 29.02791   ,
               10.334898  ,  0.        ],
              [ 0.        , 18.2472    ,  9.884217  , ..., 13.656612  ,
                7.7520328 ,  0.        ],
              [ 3.0784545 , 21.083475  ,  7.9666123 , ...,  8.91951   ,
                1.1958745 ,  0.        ],
              ...,
              [19.34507   ,  6.6602154 ,  4.9660463 , ...,  7.812122  ,
                0.        ,  0.        ],
              [22.420664  , 11.550558  ,  7.77517   , ..., 13.05619   ,
                0.        ,  0.        ],
              [14.818057  , 33.21821   ,  7.4220505 , ..., 16.502018  ,
                0.        ,  0.        ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 28, 28, 256), dtype=float32, numpy=
     array([[[[13.905667  , 58.035545  , 15.849893  , ..., 30.157932  ,
                0.        ,  6.9715033 ],
              [23.427223  , 58.147404  , 11.471322  , ...,  0.        ,
                0.        ,  8.5846615 ],
              [35.90345   , 58.27013   , 15.1265955 , ...,  6.806363  ,
                0.        ,  7.400784  ],
              ...,
              [23.005981  , 58.22731   , 18.102438  , ..., 10.173488  ,
                0.        ,  6.512558  ],
              [25.246674  , 55.223343  , 21.483719  , ...,  7.415467  ,
                0.        ,  3.6192179 ],
              [21.809414  , 84.87999   , 25.393993  , ...,  6.1818347 ,
               10.2836685 ,  4.883298  ]],
     
             [[17.230446  , 48.221222  , 13.598929  , ..., 38.382908  ,
                0.        ,  4.728796  ],
              [ 8.421812  , 33.45469   ,  1.3071035 , ...,  0.        ,
                0.        ,  5.8190064 ],
              [13.541994  , 32.256466  ,  4.38139   , ...,  0.        ,
                0.        ,  0.        ],
              ...,
              [10.0681    , 33.944088  , 10.713256  , ...,  6.175546  ,
                0.        ,  6.74692   ],
              [35.046417  , 26.267101  ,  8.801834  , ...,  0.        ,
                0.        ,  9.017897  ],
              [37.420162  , 77.83233   , 15.3065195 , ..., 14.2165365 ,
                0.        ,  9.161784  ]],
     
             [[20.35227   , 44.983295  , 13.244948  , ..., 32.857346  ,
                8.903593  ,  2.2526577 ],
              [13.06105   , 30.698374  ,  6.18084   , ...,  0.        ,
                1.7017744 ,  8.975825  ],
              [ 8.770241  , 34.078606  ,  0.60933894, ...,  0.        ,
                0.        ,  9.12513   ],
              ...,
              [15.566159  , 34.333588  ,  3.3374088 , ...,  4.833857  ,
                0.        ,  2.4518201 ],
              [21.201363  , 33.79702   ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [35.588207  , 68.4666    ,  6.4274364 , ..., 17.357067  ,
                0.        ,  4.1784897 ]],
     
             ...,
     
             [[ 0.        , 51.619164  , 11.271819  , ..., 26.520798  ,
                4.3144374 ,  3.7839782 ],
              [ 0.        , 36.82892   ,  5.4070706 , ...,  3.2318416 ,
                0.        ,  2.2386715 ],
              [ 0.        , 32.99035   ,  2.422752  , ...,  4.9013166 ,
                0.        ,  2.951623  ],
              ...,
              [19.081022  , 32.450344  ,  6.5473495 , ..., 27.520424  ,
                0.        ,  3.4021719 ],
              [ 5.6284466 , 28.609125  ,  9.383961  , ...,  6.713073  ,
                0.        ,  1.3227849 ],
              [ 6.098253  , 70.07725   , 15.384129  , ..., 20.671085  ,
                0.        ,  1.3875155 ]],
     
             [[ 8.220697  , 46.90642   ,  7.9303365 , ..., 27.404055  ,
                0.        ,  1.2637705 ],
              [10.083044  , 37.81973   ,  0.74569106, ...,  3.8368309 ,
                0.        ,  0.16889971],
              [ 5.760722  , 40.244915  ,  0.        , ...,  4.2234516 ,
                0.        ,  2.0256605 ],
              ...,
              [17.967386  , 41.520092  ,  7.5797157 , ..., 24.182344  ,
                0.        , 15.712267  ],
              [32.99098   , 36.1311    ,  6.905347  , ...,  9.685187  ,
                0.        , 11.229238  ],
              [36.0115    , 75.41097   , 15.488862  , ..., 23.293133  ,
                0.        ,  0.        ]],
     
             [[13.79315   , 47.67468   , 12.265128  , ..., 29.02791   ,
               10.334898  ,  2.8166153 ],
              [23.337914  , 49.30081   ,  9.79031   , ..., 15.457357  ,
                1.1958745 ,  0.11377796],
              [16.110992  , 52.97133   , 10.983256  , ..., 20.206896  ,
                1.6933204 ,  0.        ],
              ...,
              [28.685333  , 42.475525  , 19.824123  , ..., 19.670744  ,
                7.216317  , 12.056595  ],
              [46.747837  , 33.25263   ,  9.408773  , ..., 10.913831  ,
                1.4190983 ,  8.1573715 ],
              [51.25247   , 71.787285  ,  7.77517   , ..., 21.104273  ,
                0.        ,  1.0477773 ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 28, 28, 512), dtype=float32, numpy=
     array([[[[ 9.559978  ,  1.2684138 ,  0.        , ...,  0.        ,
               46.573624  ,  0.        ],
              [12.11866   , 10.297334  ,  0.        , ...,  0.        ,
               67.29535   ,  0.        ],
              [14.6669655 , 20.817148  ,  0.        , ...,  0.        ,
               70.595695  ,  0.        ],
              ...,
              [ 9.572271  , 20.634687  ,  0.        , ...,  0.        ,
               56.43752   ,  0.        ],
              [ 7.6578937 , 20.759335  ,  0.        , ...,  0.        ,
               51.260662  ,  0.        ],
              [10.5464115 , 15.21024   ,  8.207669  , ...,  0.        ,
               40.903     ,  0.        ]],
     
             [[ 7.922499  , 19.824408  ,  0.        , ...,  0.        ,
               37.295387  ,  0.        ],
              [ 0.        , 36.96139   ,  3.172225  , ...,  0.        ,
               32.94567   ,  0.        ],
              [ 0.        , 34.82676   , 14.411455  , ...,  0.        ,
               17.913904  ,  0.        ],
              ...,
              [ 0.        , 30.656439  ,  4.11961   , ...,  0.        ,
               21.499926  ,  0.        ],
              [ 0.        , 32.911358  ,  2.3964071 , ...,  0.        ,
               21.909472  ,  0.        ],
              [10.155024  , 28.64786   , 30.37974   , ...,  0.        ,
               23.352232  ,  0.        ]],
     
             [[ 4.368649  , 26.155264  ,  0.        , ...,  0.        ,
               12.709506  ,  4.586512  ],
              [ 0.        , 44.593414  ,  0.        , ...,  0.        ,
                6.982811  ,  0.        ],
              [ 0.        , 40.605743  ,  6.230919  , ...,  0.        ,
                9.815583  ,  0.        ],
              ...,
              [ 0.        , 34.404392  ,  0.        , ...,  0.        ,
               26.37444   ,  0.        ],
              [ 0.        , 29.193079  ,  0.        , ...,  0.        ,
               19.522638  ,  0.        ],
              [ 3.9019191 , 23.109264  , 19.733662  , ...,  0.        ,
               16.764277  ,  0.        ]],
     
             ...,
     
             [[10.913452  , 14.78303   ,  0.        , ...,  0.        ,
               10.340772  , 10.663865  ],
              [ 5.651646  , 31.521063  ,  0.        , ...,  0.        ,
                8.455928  ,  0.        ],
              [ 2.8179238 , 37.5562    ,  0.        , ...,  0.        ,
               19.763945  ,  0.        ],
              ...,
              [ 3.8808906 , 26.130548  ,  0.        , ...,  0.        ,
               20.568937  ,  0.        ],
              [ 0.        , 30.836689  ,  0.        , ...,  0.        ,
               19.643427  ,  0.        ],
              [ 0.        , 34.838875  ,  2.01407   , ...,  0.        ,
               20.724586  ,  0.        ]],
     
             [[10.030939  ,  8.580132  ,  0.        , ...,  0.        ,
                3.0627048 ,  7.595047  ],
              [13.240556  , 25.95319   ,  0.        , ...,  0.        ,
                7.810996  ,  0.        ],
              [11.815131  , 35.919426  ,  0.        , ...,  0.        ,
               22.118738  ,  0.        ],
              ...,
              [ 0.        , 29.43942   ,  0.        , ...,  0.        ,
               29.549911  ,  0.        ],
              [ 0.        , 27.918571  ,  0.        , ...,  0.        ,
               17.851778  ,  0.        ],
              [ 0.        , 26.107862  , 15.245581  , ...,  0.        ,
               18.323868  ,  0.        ]],
     
             [[ 2.80882   , 34.587955  ,  0.24607086, ...,  0.        ,
                0.        , 19.591711  ],
              [ 8.018656  , 55.83762   , 16.74941   , ...,  0.        ,
                0.        , 14.281405  ],
              [11.554534  , 54.83378   , 14.025097  , ...,  0.        ,
                0.        , 14.86191   ],
              ...,
              [ 0.        , 51.311962  , 19.060452  , ...,  0.        ,
                0.        ,  7.656452  ],
              [ 0.        , 48.205967  , 24.170021  , ...,  0.        ,
                0.        , 17.208471  ],
              [14.269529  , 34.215     , 35.453236  , ...,  0.        ,
                0.        , 19.722212  ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 28, 28, 512), dtype=float32, numpy=
     array([[[[11.154971 ,  0.       ,  0.       , ...,  7.692477 ,
               11.352772 ,  0.       ],
              [ 0.       ,  0.       ,  0.       , ...,  2.5605392,
                4.8575478,  0.       ],
              [ 0.       ,  0.       ,  0.       , ...,  1.4104104,
               13.323062 ,  1.4653636],
              ...,
              [ 0.       ,  0.       ,  0.       , ...,  6.606435 ,
               11.438259 ,  0.       ],
              [ 0.       ,  0.       ,  0.       , ...,  0.       ,
               15.312402 ,  0.7722526],
              [ 0.       ,  0.       ,  3.5149746, ...,  8.245947 ,
                9.89737  , 12.76327  ]],
     
             [[ 0.       ,  0.       ,  0.       , ..., 23.245533 ,
                7.722185 ,  0.       ],
              [ 0.       ,  0.       ,  0.       , ..., 27.805866 ,
                0.       ,  4.4071755],
              [ 0.       ,  0.       ,  0.       , ..., 23.588484 ,
                7.7223787, 12.329492 ],
              ...,
              [ 0.       ,  0.       ,  0.       , ..., 24.187965 ,
                0.       ,  0.       ],
              [ 0.       ,  0.       ,  0.       , ..., 23.36277  ,
                4.9730253,  0.       ],
              [ 0.       ,  0.       ,  0.3027315, ..., 24.602392 ,
                3.3250842,  6.1685376]],
     
             [[ 0.       ,  0.       ,  0.       , ..., 23.651789 ,
                4.679467 ,  3.352275 ],
              [ 0.       ,  0.       ,  0.       , ..., 32.119072 ,
                0.       ,  3.094619 ],
              [ 0.       ,  0.       ,  0.       , ..., 30.594486 ,
                0.       ,  5.8671017],
              ...,
              [ 0.       ,  0.       ,  0.       , ..., 20.164545 ,
                0.       ,  0.       ],
              [ 0.       ,  0.       ,  0.       , ..., 37.333874 ,
                0.       ,  0.       ],
              [ 0.       ,  0.       ,  0.3330136, ..., 38.708298 ,
                5.1087728,  5.676172 ]],
     
             ...,
     
             [[ 0.       ,  0.       ,  0.       , ..., 25.01116  ,
                0.       ,  0.       ],
              [ 0.       ,  0.       ,  0.       , ..., 23.36327  ,
                0.       ,  0.       ],
              [ 0.       ,  0.       ,  0.       , ..., 26.55672  ,
                0.       ,  0.       ],
              ...,
              [ 0.       ,  0.       ,  0.       , ...,  5.49049  ,
               20.024677 ,  0.       ],
              [ 0.       ,  0.       ,  0.       , ..., 14.234548 ,
                0.       ,  0.       ],
              [ 0.       ,  0.       ,  0.       , ..., 37.727295 ,
               12.726037 ,  6.3723035]],
     
             [[ 0.       ,  0.       ,  0.       , ..., 21.632257 ,
                3.3110428,  0.       ],
              [ 0.       ,  0.       ,  0.       , ..., 27.610235 ,
                0.       ,  0.       ],
              [ 0.       ,  0.       ,  0.       , ..., 32.137894 ,
                1.2463177,  0.       ],
              ...,
              [ 0.       ,  0.       ,  0.       , ...,  0.       ,
               28.248802 ,  0.       ],
              [ 0.       ,  0.       ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       ,  0.       ,  0.       , ..., 15.334745 ,
                9.692297 ,  3.5858078]],
     
             [[ 0.       ,  0.       ,  0.       , ...,  5.0302806,
                5.47307  ,  0.       ],
              [ 0.       ,  0.       ,  0.       , ...,  9.929971 ,
                7.5017443,  0.       ],
              [ 0.       ,  0.       ,  0.       , ...,  7.3860483,
                9.714461 ,  0.       ],
              ...,
              [ 0.       ,  0.       ,  0.       , ...,  0.       ,
               24.3265   ,  0.       ],
              [ 0.       ,  0.       ,  0.       , ...,  0.       ,
                2.3154013,  0.       ],
              [ 0.       ,  0.       ,  0.       , ...,  9.474858 ,
                9.783551 ,  5.413635 ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 28, 28, 512), dtype=float32, numpy=
     array([[[[ 0.       ,  5.4787073,  4.4998155, ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       ,  0.       ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       ,  1.7949266,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              ...,
              [ 0.       ,  0.       ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       ,  0.       ,  6.5409746, ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       ,  0.       , 16.08492  , ...,  0.       ,
                2.0800922,  0.       ]],
     
             [[ 0.       , 30.40721  ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       , 32.80215  ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       , 37.79688  ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              ...,
              [ 0.       , 25.584723 ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       , 33.979725 ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       , 35.363075 , 14.450548 , ...,  0.       ,
                0.       ,  0.       ]],
     
             [[ 0.       , 26.03385  ,  0.       , ...,  0.       ,
                1.6739737,  0.       ],
              [ 0.       , 27.770296 ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       , 28.885136 ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              ...,
              [ 0.       , 21.948494 ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       , 32.164825 ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       , 39.758743 ,  9.813249 , ...,  0.       ,
                0.       ,  0.       ]],
     
             ...,
     
             [[ 0.       , 24.05923  ,  0.       , ...,  0.       ,
                9.313571 ,  0.       ],
              [ 0.       , 19.170113 ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       , 25.117983 ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              ...,
              [ 0.       , 17.9292   ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       , 31.555735 ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       , 36.56331  ,  7.6627517, ...,  0.       ,
                0.       ,  0.       ]],
     
             [[ 0.       , 29.118912 ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       , 20.988775 ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       , 26.99918  ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              ...,
              [ 0.       , 19.73042  ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       , 33.853806 ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       , 40.72294  , 13.127812 , ...,  0.       ,
                0.       ,  0.       ]],
     
             [[ 0.       , 28.922985 ,  0.       , ...,  0.       ,
               15.872481 ,  0.       ],
              [ 0.       , 28.789568 ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              [ 0.       , 21.495718 ,  0.       , ...,  0.       ,
                0.       ,  0.       ],
              ...,
              [ 0.       , 23.606966 ,  0.       , ...,  0.       ,
                4.791225 ,  0.       ],
              [ 0.       , 32.01571  ,  0.       , ...,  0.       ,
                4.843074 ,  0.       ],
              [ 0.       , 41.00196  ,  3.1229875, ...,  0.       ,
               10.976443 ,  2.4694996]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 28, 28, 512), dtype=float32, numpy=
     array([[[[ 0.        ,  9.237843  ,  0.        , ...,  0.        ,
                0.9105259 ,  0.        ],
              [ 0.        ,  7.22767   ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  4.5843625 ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              ...,
              [ 0.        ,  8.023155  ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  5.6740236 ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  8.216747  ,  0.        , ...,  0.        ,
                0.        ,  0.        ]],
     
             [[ 0.        ,  5.793042  ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  6.8725686 ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  6.967202  ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              ...,
              [ 0.        ,  2.5082498 ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 6.6846952 ,  5.670997  ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 3.700808  , 17.5017    ,  0.        , ...,  0.        ,
                0.        ,  0.        ]],
     
             [[ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              ...,
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 4.6308193 ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 1.3957973 ,  4.251856  ,  0.        , ...,  0.        ,
                0.        ,  0.        ]],
     
             ...,
     
             [[ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.03235737, ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  2.5183835 , ...,  0.        ,
                0.        ,  0.        ],
              ...,
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 3.7052617 ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  8.110795  ,  0.        , ...,  0.        ,
                0.        ,  0.        ]],
     
             [[ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              ...,
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  2.8297756 ,  0.        , ...,  0.        ,
                0.        ,  0.        ]],
     
             [[ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              ...,
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 3.3674638 ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 1.8639882 ,  0.        ,  0.        , ...,  0.        ,
                4.238995  ,  0.        ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 14, 14, 512), dtype=float32, numpy=
     array([[[[ 0.        ,  9.237843  ,  0.        , ...,  0.        ,
                0.9105259 ,  0.        ],
              [ 0.        ,  6.967202  ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  6.2922444 ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              ...,
              [ 0.        ,  6.89842   ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  8.0669775 ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 6.6846952 , 17.5017    ,  0.        , ...,  0.        ,
                0.        ,  0.        ]],
     
             [[ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                1.5542777 ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              ...,
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                1.4228716 ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 4.6308193 ,  4.251856  ,  0.        , ...,  0.        ,
                0.        ,  0.        ]],
     
             [[ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              ...,
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ]],
     
             ...,
     
             [[ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  1.4066186 , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              ...,
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.6760854 ,  0.        , ...,  0.        ,
                0.        ,  0.        ]],
     
             [[ 0.        ,  0.        ,  0.03235737, ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  2.5183835 , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              ...,
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 3.7052617 ,  8.110795  ,  0.        , ...,  0.        ,
                0.        ,  0.        ]],
     
             [[ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              ...,
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 0.        ,  0.        ,  0.        , ...,  0.        ,
                0.        ,  0.        ],
              [ 3.3674638 ,  2.8297756 ,  0.        , ...,  0.        ,
                4.238995  ,  0.        ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 14, 14, 512), dtype=float32, numpy=
     array([[[[0.5012002 , 0.        , 3.4602735 , ..., 1.4485227 ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.20601325,
               0.        , 0.        ],
              ...,
              [0.        , 0.        , 0.        , ..., 0.31216684,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 1.7421104 ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ]],
     
             [[0.93172365, 0.        , 3.582056  , ..., 0.27604616,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              ...,
              [0.        , 0.        , 0.        , ..., 0.07283132,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 2.1067836 ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ]],
     
             [[1.1142364 , 0.        , 0.81597084, ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              ...,
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 1.076424  ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ]],
     
             ...,
     
             [[1.0145807 , 0.        , 1.7624173 , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              ...,
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ]],
     
             [[0.        , 0.        , 4.350695  , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              ...,
              [0.        , 0.        , 0.        , ..., 1.1734529 ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 1.7764236 ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ]],
     
             [[2.7599    , 0.        , 6.3250446 , ..., 0.16521882,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              ...,
              [0.        , 0.        , 0.5336036 , ..., 1.4294426 ,
               0.        , 0.        ],
              [0.        , 0.        , 0.5560287 , ..., 2.6390836 ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 14, 14, 512), dtype=float32, numpy=
     array([[[[0.        , 0.        , 3.730213  , ..., 0.        ,
               4.707105  , 1.1265726 ],
              [0.        , 0.        , 0.2842995 , ..., 0.        ,
               5.7049985 , 1.2672689 ],
              [0.        , 0.        , 0.19591485, ..., 0.        ,
               4.3418436 , 1.2334139 ],
              ...,
              [0.        , 0.        , 0.        , ..., 0.        ,
               4.4745417 , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               4.3782554 , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               2.9404533 , 0.07233718]],
     
             [[0.        , 0.        , 3.4818296 , ..., 0.        ,
               0.        , 1.0813931 ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.20971814, 0.7479219 ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.26517326, 0.5308557 ],
              ...,
              [0.        , 0.        , 0.        , ..., 0.        ,
               1.7649409 , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.68683326, 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ]],
     
             [[0.        , 0.        , 3.1756628 , ..., 0.43465805,
               0.41099653, 1.063434  ],
              [0.        , 0.        , 0.        , ..., 0.16166314,
               0.5014306 , 0.906362  ],
              [0.        , 0.        , 0.        , ..., 0.81646204,
               0.46575803, 0.7199232 ],
              ...,
              [0.        , 0.        , 0.        , ..., 1.7824546 ,
               1.7180593 , 0.        ],
              [0.        , 0.        , 0.        , ..., 1.0410725 ,
               1.0136609 , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.3396485 ,
               0.        , 0.34325796]],
     
             ...,
     
             [[0.        , 0.        , 3.1573176 , ..., 0.        ,
               0.        , 1.4526777 ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 1.8215716 ],
              [0.        , 0.        , 0.        , ..., 0.22885698,
               0.        , 1.8508445 ],
              ...,
              [0.        , 0.        , 0.        , ..., 1.15934   ,
               0.09874126, 0.5626127 ],
              [0.        , 0.        , 0.        , ..., 0.3209104 ,
               0.22919905, 0.13269335],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.6129491 , 0.79405755]],
     
             [[0.        , 0.        , 3.3119094 , ..., 0.        ,
               0.        , 1.5826825 ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 1.3803816 ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 1.2913549 ],
              ...,
              [0.        , 0.        , 0.        , ..., 0.397314  ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.3641789 , 0.        ]],
     
             [[0.        , 0.        , 3.29589   , ..., 0.0526222 ,
               0.        , 1.6205945 ],
              [0.        , 0.        , 0.16975941, ..., 0.07270789,
               0.        , 1.402762  ],
              [0.        , 0.        , 0.04170351, ..., 0.733566  ,
               0.        , 1.1284004 ],
              ...,
              [0.        , 0.        , 0.12836799, ..., 0.48960942,
               0.        , 0.23677191],
              [0.        , 0.        , 0.46826977, ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.629997  , ..., 0.        ,
               0.        , 0.        ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 14, 14, 512), dtype=float32, numpy=
     array([[[[0.        , 0.        , 0.        , ..., 0.28385085,
               0.        , 0.28486848],
              [0.        , 0.        , 0.        , ..., 0.09540667,
               0.        , 0.46559507],
              [0.        , 0.        , 0.06469972, ..., 0.11326183,
               0.        , 0.25219935],
              ...,
              [0.        , 0.        , 0.15046412, ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.22678858, ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.5048795 , 0.        ]],
     
             [[0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.38608766],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.8455851 ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.34699696],
              ...,
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.30934456, 0.34265357]],
     
             [[0.        , 0.        , 0.68686026, ..., 0.        ,
               0.        , 0.3413805 ],
              [0.        , 0.        , 0.22820458, ..., 0.        ,
               0.        , 0.3046166 ],
              [0.        , 0.        , 0.7452977 , ..., 0.        ,
               0.        , 0.        ],
              ...,
              [0.        , 0.        , 0.5255914 , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.62860143, ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.02612588, ..., 0.        ,
               0.        , 0.        ]],
     
             ...,
     
             [[0.        , 0.        , 0.80489683, ..., 0.        ,
               0.        , 0.08949147],
              [0.        , 0.        , 0.3555476 , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.8579501 , ..., 0.        ,
               0.        , 0.        ],
              ...,
              [0.        , 0.        , 0.5471438 , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.8835474 , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.15601169, ..., 0.        ,
               0.        , 0.        ]],
     
             [[0.        , 0.        , 0.9814981 , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.7217578 , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 1.1234758 , ..., 0.        ,
               0.        , 0.        ],
              ...,
              [0.        , 0.        , 0.8150461 , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 1.1837366 , ..., 0.        ,
               0.        , 0.        ],
              [0.        , 0.        , 0.29467297, ..., 0.        ,
               0.07879904, 0.        ]],
     
             [[0.        , 0.        , 0.42021665, ..., 0.44609478,
               0.39998877, 0.2119496 ],
              [0.        , 0.        , 0.        , ..., 0.01309705,
               0.48399654, 0.5068003 ],
              [0.        , 0.        , 0.26097542, ..., 0.        ,
               0.1850368 , 0.        ],
              ...,
              [0.00630613, 0.        , 0.        , ..., 0.12620795,
               0.07227121, 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.49912068, 0.        ],
              [0.        , 0.        , 0.        , ..., 0.01374803,
               0.80786276, 0.17238696]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 14, 14, 512), dtype=float32, numpy=
     array([[[[0.38522702, 0.        , 0.        , ..., 0.        ,
               0.9529687 , 0.        ],
              [0.51641095, 0.        , 0.        , ..., 0.        ,
               0.4878134 , 0.        ],
              [0.5697699 , 0.        , 0.        , ..., 0.        ,
               0.577434  , 0.        ],
              ...,
              [0.5943756 , 0.        , 0.        , ..., 0.        ,
               0.97988904, 0.        ],
              [0.6934797 , 0.        , 0.        , ..., 0.        ,
               0.8374356 , 0.        ],
              [0.48094442, 0.        , 0.        , ..., 0.        ,
               0.85279727, 0.        ]],
     
             [[0.        , 0.        , 0.        , ..., 0.        ,
               0.90378547, 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.26113087, 0.        ],
              [0.03523093, 0.        , 0.        , ..., 0.        ,
               0.33739886, 0.        ],
              ...,
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.6425067 , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.41176897, 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.6340584 , 0.        ]],
     
             [[0.07161272, 0.        , 0.        , ..., 0.        ,
               1.1840122 , 0.        ],
              [0.19266397, 0.        , 0.        , ..., 0.        ,
               0.61674774, 0.        ],
              [0.2397702 , 0.        , 0.        , ..., 0.        ,
               0.67872804, 0.        ],
              ...,
              [0.21977878, 0.        , 0.        , ..., 0.        ,
               0.9357337 , 0.        ],
              [0.4017149 , 0.        , 0.        , ..., 0.        ,
               0.77797633, 0.        ],
              [0.2753923 , 0.        , 0.        , ..., 0.        ,
               0.85565466, 0.        ]],
     
             ...,
     
             [[0.01096076, 0.        , 0.        , ..., 0.        ,
               1.2763412 , 0.        ],
              [0.08173943, 0.        , 0.        , ..., 0.        ,
               0.70492095, 0.        ],
              [0.03767866, 0.        , 0.        , ..., 0.        ,
               0.81114656, 0.        ],
              ...,
              [0.02379602, 0.        , 0.        , ..., 0.        ,
               0.9223137 , 0.        ],
              [0.15301287, 0.        , 0.        , ..., 0.        ,
               0.75982153, 0.        ],
              [0.10222512, 0.        , 0.        , ..., 0.        ,
               0.8155729 , 0.        ]],
     
             [[0.        , 0.        , 0.        , ..., 0.        ,
               1.2119321 , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.6789073 , 0.        ],
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.91906965, 0.        ],
              ...,
              [0.        , 0.        , 0.        , ..., 0.        ,
               0.80133265, 0.        ],
              [0.0318203 , 0.        , 0.        , ..., 0.        ,
               0.6014866 , 0.        ],
              [0.04225051, 0.        , 0.        , ..., 0.        ,
               0.67331886, 0.        ]],
     
             [[0.12846619, 0.        , 0.        , ..., 0.        ,
               1.2850423 , 0.        ],
              [0.05461901, 0.        , 0.        , ..., 0.        ,
               0.9956109 , 0.        ],
              [0.18795955, 0.        , 0.        , ..., 0.        ,
               1.0908533 , 0.        ],
              ...,
              [0.4077804 , 0.        , 0.        , ..., 0.        ,
               0.7675604 , 0.        ],
              [0.4288964 , 0.        , 0.        , ..., 0.        ,
               0.65767944, 0.        ],
              [0.36514688, 0.        , 0.        , ..., 0.        ,
               0.7133655 , 0.        ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 7, 7, 512), dtype=float32, numpy=
     array([[[[0.51641095, 0.        , 0.        , ..., 0.        ,
               0.9529687 , 0.        ],
              [0.7705579 , 0.        , 0.        , ..., 0.        ,
               0.607028  , 0.        ],
              [0.7847327 , 0.        , 0.01307762, ..., 0.        ,
               0.72942686, 0.        ],
              ...,
              [0.7474328 , 0.        , 0.        , ..., 0.        ,
               0.7308583 , 0.        ],
              [0.7433806 , 0.        , 0.        , ..., 0.        ,
               0.97988904, 0.        ],
              [0.6934797 , 0.        , 0.        , ..., 0.        ,
               0.85279727, 0.        ]],
     
             [[0.19266397, 0.        , 0.        , ..., 0.        ,
               1.3493984 , 0.        ],
              [0.3930894 , 0.        , 0.        , ..., 0.        ,
               0.7525735 , 0.        ],
              [0.45968243, 0.        , 0.        , ..., 0.        ,
               0.83063114, 0.        ],
              ...,
              [0.39060855, 0.        , 0.        , ..., 0.        ,
               0.82970375, 0.        ],
              [0.38296962, 0.        , 0.        , ..., 0.        ,
               0.9357337 , 0.        ],
              [0.4017149 , 0.        , 0.        , ..., 0.        ,
               0.87667   , 0.        ]],
     
             [[0.27289674, 0.        , 0.        , ..., 0.        ,
               1.3632486 , 0.        ],
              [0.26704523, 0.        , 0.        , ..., 0.        ,
               0.75682026, 0.        ],
              [0.3476232 , 0.        , 0.        , ..., 0.        ,
               0.88666   , 0.        ],
              ...,
              [0.43353182, 0.        , 0.        , ..., 0.        ,
               0.78889525, 0.        ],
              [0.41809207, 0.        , 0.        , ..., 0.        ,
               0.7911395 , 0.        ],
              [0.3640357 , 0.        , 0.        , ..., 0.        ,
               0.8832263 , 0.        ]],
     
             ...,
     
             [[0.25787175, 0.        , 0.        , ..., 0.        ,
               1.2768645 , 0.        ],
              [0.2949237 , 0.        , 0.        , ..., 0.        ,
               0.7651787 , 0.        ],
              [0.4413415 , 0.        , 0.        , ..., 0.        ,
               0.90104115, 0.        ],
              ...,
              [0.49801588, 0.        , 0.        , ..., 0.        ,
               0.87795043, 0.        ],
              [0.4627564 , 0.        , 0.        , ..., 0.        ,
               0.8140259 , 0.        ],
              [0.33228582, 0.        , 0.        , ..., 0.        ,
               0.83095336, 0.        ]],
     
             [[0.14356595, 0.        , 0.        , ..., 0.        ,
               1.2763412 , 0.        ],
              [0.2578581 , 0.        , 0.        , ..., 0.        ,
               0.81114656, 0.        ],
              [0.41077885, 0.        , 0.        , ..., 0.        ,
               0.92047566, 0.        ],
              ...,
              [0.49159786, 0.        , 0.        , ..., 0.        ,
               0.9823118 , 0.        ],
              [0.462144  , 0.        , 0.        , ..., 0.        ,
               0.9223137 , 0.        ],
              [0.3471186 , 0.        , 0.        , ..., 0.        ,
               0.8155729 , 0.        ]],
     
             [[0.12846619, 0.        , 0.        , ..., 0.        ,
               1.2850423 , 0.        ],
              [0.30775973, 0.        , 0.        , ..., 0.        ,
               1.0908533 , 0.        ],
              [0.40994254, 0.        , 0.        , ..., 0.        ,
               0.9444089 , 0.        ],
              ...,
              [0.54165715, 0.        , 0.        , ..., 0.        ,
               0.97811806, 0.        ],
              [0.5102333 , 0.        , 0.        , ..., 0.        ,
               0.84064287, 0.        ],
              [0.4288964 , 0.        , 0.        , ..., 0.        ,
               0.7133655 , 0.        ]]]], dtype=float32)>,
     <tf.Tensor: shape=(1, 25088), dtype=float32, numpy=
     array([[0.51641095, 0.        , 0.        , ..., 0.        , 0.7133655 ,
             0.        ]], dtype=float32)>,
     <tf.Tensor: shape=(1, 4096), dtype=float32, numpy=
     array([[0.       , 1.7807641, 0.       , ..., 0.       , 2.822582 ,
             1.2279682]], dtype=float32)>,
     <tf.Tensor: shape=(1, 4096), dtype=float32, numpy=
     array([[0.54855525, 0.        , 0.384629  , ..., 0.        , 0.6007352 ,
             0.        ]], dtype=float32)>,
     <tf.Tensor: shape=(1, 1000), dtype=float32, numpy=
     array([[1.99950897e-04, 1.37428369e-03, 6.93583570e-04, 1.64656108e-03,
             2.39033601e-03, 2.17322237e-03, 8.60171020e-03, 1.86499965e-04,
             2.81209243e-04, 2.34975698e-04, 6.66218926e-04, 2.94205907e-04,
             4.01223515e-04, 8.57901352e-04, 3.57034645e-04, 3.44276399e-04,
             7.07466505e-04, 3.85694322e-04, 7.26314669e-04, 1.08506018e-03,
             1.47488224e-03, 9.28875583e-04, 6.84596249e-04, 5.75523649e-04,
             1.36582879e-04, 2.63555499e-04, 1.21544767e-03, 1.17233582e-03,
             2.87497824e-04, 4.90275491e-03, 3.90844856e-04, 4.43229394e-04,
             4.59499191e-04, 1.57884846e-03, 1.02192047e-03, 5.55509352e-04,
             1.32443861e-03, 4.12939582e-04, 2.61690863e-03, 3.00602143e-04,
             8.02435679e-04, 7.36410671e-04, 1.12145813e-03, 5.81005996e-04,
             9.32326482e-04, 2.07000296e-03, 9.82223428e-04, 7.37288909e-04,
             5.59532782e-04, 5.69262775e-04, 9.35597578e-04, 3.45145119e-04,
             2.79695843e-03, 2.79069436e-03, 8.60725937e-04, 3.26874724e-04,
             1.51876884e-03, 2.36233289e-04, 1.12430723e-02, 4.05299652e-04,
             1.35547959e-03, 1.09705830e-03, 1.07030279e-03, 9.54928808e-04,
             2.63193552e-03, 1.64226815e-03, 2.68660393e-03, 1.08009391e-03,
             1.28068402e-03, 2.42663338e-03, 1.59104448e-03, 2.34784861e-03,
             6.23393629e-04, 1.15435373e-03, 5.06066601e-04, 4.95459326e-03,
             1.43369404e-03, 2.19900114e-03, 9.19999089e-03, 5.06639201e-03,
             1.10668328e-03, 4.87727346e-03, 6.15183264e-04, 7.19749427e-04,
             3.03175999e-04, 1.76315755e-03, 1.61445013e-03, 2.13475875e-03,
             2.52152589e-04, 2.55966978e-03, 8.11416830e-05, 3.98395350e-04,
             9.93116410e-04, 1.75176203e-04, 7.77936017e-04, 1.57656643e-04,
             4.48439532e-04, 2.52028083e-04, 9.22670646e-04, 7.00403703e-04,
             2.87463103e-04, 2.55216146e-04, 3.25172587e-04, 1.80152466e-03,
             2.71619821e-04, 9.13662225e-05, 6.29426213e-04, 6.45179185e-04,
             5.82505716e-04, 5.79487125e-04, 2.04615775e-04, 3.10132583e-03,
             1.30564650e-03, 5.61843917e-04, 6.18987600e-04, 7.79826005e-05,
             1.79716459e-04, 1.54949888e-03, 3.75883130e-04, 2.46054056e-04,
             2.28807027e-03, 7.88223042e-05, 1.09788241e-04, 9.57667289e-05,
             1.31733518e-03, 1.40024605e-03, 2.03092699e-03, 9.79409087e-04,
             1.11713447e-03, 9.24865133e-04, 2.42108275e-04, 8.54907033e-04,
             2.06752494e-03, 3.27342015e-04, 8.12767888e-04, 2.69906159e-04,
             1.81767813e-04, 1.19874242e-03, 2.54714536e-03, 6.47519366e-04,
             1.79868261e-03, 1.92794111e-03, 1.49870303e-03, 8.71106517e-04,
             4.70825384e-04, 6.86897722e-04, 1.86268822e-03, 1.35011462e-04,
             1.87862461e-04, 2.13656086e-03, 7.11464265e-04, 1.35362463e-03,
             2.80075241e-04, 4.18291893e-03, 1.86037412e-03, 1.36173703e-03,
             6.68334833e-04, 9.44758824e-04, 5.66532253e-04, 2.19776659e-04,
             2.60044151e-04, 6.47429551e-04, 1.34245830e-03, 5.37992339e-04,
             3.37319652e-04, 9.37209043e-05, 7.31378270e-04, 1.67793492e-04,
             4.31404449e-04, 4.03854880e-04, 5.96055237e-04, 1.36940670e-03,
             1.73750916e-03, 6.91822148e-04, 4.65190678e-04, 1.41474535e-04,
             3.23202083e-04, 1.98117195e-04, 1.04176626e-03, 7.94861407e-04,
             7.52831285e-04, 6.71321177e-04, 5.37213171e-04, 8.06786702e-05,
             4.20958328e-04, 4.18239331e-04, 3.05509078e-04, 2.66536488e-04,
             3.36063531e-04, 2.73957674e-04, 1.05668197e-03, 1.65197212e-04,
             1.15026592e-03, 2.30721780e-04, 3.72814829e-04, 4.44940408e-04,
             5.08681813e-04, 1.11097543e-04, 9.44752464e-05, 6.40808721e-04,
             2.24004645e-04, 1.46952007e-04, 9.05173714e-04, 4.05651657e-03,
             1.76043983e-03, 8.56509723e-05, 6.89507287e-05, 6.11459895e-04,
             6.29709626e-04, 1.55166595e-03, 2.33052066e-04, 3.07678041e-04,
             8.04436102e-04, 1.48562685e-04, 1.27640873e-04, 6.88912638e-04,
             3.41843290e-04, 5.09614416e-04, 4.51951317e-04, 2.56548054e-04,
             1.03512830e-04, 6.98226359e-05, 1.50630902e-03, 3.76752345e-04,
             8.54268874e-05, 2.52902420e-04, 1.73765409e-04, 1.77340538e-04,
             3.83779407e-04, 1.02936127e-03, 5.95491612e-04, 2.46481854e-04,
             2.45647971e-04, 1.65468722e-04, 1.92337102e-04, 1.37982235e-04,
             2.73019250e-04, 1.05051114e-03, 5.59483713e-04, 4.46740654e-04,
             1.88746271e-04, 2.49542412e-04, 9.96469054e-04, 2.96994345e-04,
             6.86450148e-05, 1.31537754e-03, 3.13321245e-04, 3.73268442e-04,
             4.65031684e-04, 4.86494682e-04, 1.35169120e-03, 5.26070595e-04,
             1.26867948e-04, 5.25200216e-04, 8.34136619e-04, 8.11165737e-05,
             1.23902573e-04, 2.45562824e-03, 6.30494719e-03, 1.21219025e-03,
             3.91304726e-04, 3.18598351e-04, 1.07539963e-04, 1.22014340e-03,
             5.03738120e-04, 1.53155765e-03, 5.58202737e-04, 2.47995835e-04,
             3.82233586e-04, 2.14083935e-04, 1.11232139e-03, 1.28242100e-04,
             4.45121521e-04, 9.23353713e-04, 3.33421573e-04, 2.31305385e-04,
             5.75126847e-04, 5.50041208e-04, 4.20452154e-04, 4.15433897e-03,
             3.84747400e-04, 4.45633545e-04, 6.64186664e-04, 3.11067700e-03,
             3.44219501e-03, 8.04339710e-04, 5.57719555e-04, 1.17098354e-03,
             2.66866758e-04, 5.50705299e-04, 1.04622392e-04, 3.87871551e-04,
             1.08117012e-04, 2.02451731e-04, 8.98526196e-05, 6.43084713e-05,
             1.36874372e-03, 2.61343579e-04, 8.99210689e-04, 1.17922691e-03,
             4.03068610e-04, 1.48354413e-03, 4.37324838e-04, 4.22895740e-04,
             3.21528583e-04, 9.82134952e-04, 6.10627758e-04, 5.28542325e-04,
             6.99810334e-04, 1.03208535e-04, 2.12411094e-03, 9.14463075e-04,
             1.59635569e-03, 2.03429116e-03, 9.67715774e-03, 1.20355433e-03,
             7.95064378e-04, 5.65490685e-04, 3.75048281e-03, 9.16735793e-04,
             4.73169785e-04, 8.86359121e-05, 2.01683884e-04, 5.60934714e-05,
             2.82054330e-04, 8.12499347e-05, 1.86937948e-04, 7.28435058e-04,
             2.84037145e-04, 5.80017921e-04, 1.24297931e-03, 1.82458921e-03,
             2.63330014e-03, 4.00048820e-03, 4.22705431e-04, 2.44973606e-04,
             3.54302116e-04, 3.00008454e-04, 4.83126816e-04, 3.42976746e-05,
             1.21235295e-04, 4.00073070e-04, 1.53288303e-04, 2.79881817e-04,
             4.65339224e-04, 1.37131181e-04, 3.24926892e-04, 1.14440059e-04,
             2.47923337e-04, 4.10619657e-04, 8.71463853e-04, 3.96744115e-04,
             1.86961741e-04, 4.56523820e-04, 4.44274279e-04, 1.76851274e-04,
             2.55070580e-03, 5.01598057e-04, 2.31789425e-03, 3.34567693e-03,
             6.11072930e-04, 3.66318884e-04, 9.46060696e-04, 1.30099920e-03,
             1.56131136e-04, 1.70624422e-04, 1.29019289e-04, 2.61496025e-04,
             1.55385904e-04, 7.09924934e-05, 2.03750416e-04, 3.99312034e-04,
             3.52629635e-04, 3.67832952e-04, 9.39061225e-04, 1.32321977e-04,
             3.13022581e-04, 6.32212323e-04, 3.74104333e-04, 1.16349089e-04,
             4.83563897e-04, 1.01396348e-04, 3.29248840e-04, 3.25993838e-04,
             9.00756495e-05, 2.63815513e-04, 4.34249319e-04, 1.51127955e-04,
             1.56836046e-04, 3.58934863e-04, 3.79359059e-04, 8.33170081e-04,
             7.89342375e-05, 3.08291463e-04, 5.58104948e-04, 1.04299840e-03,
             1.75475594e-04, 7.19219039e-04, 1.95264365e-04, 4.17050411e-04,
             2.06123768e-05, 1.40800927e-04, 2.21037335e-04, 8.20938440e-05,
             7.28633255e-04, 2.16434780e-03, 6.83660182e-05, 1.86469566e-04,
             5.55523475e-05, 3.57679266e-04, 8.29114942e-05, 7.69453414e-04,
             5.62339148e-04, 3.99555254e-04, 3.28926079e-04, 1.95664499e-04,
             4.27788735e-04, 1.23079924e-03, 4.22484893e-03, 2.09234878e-02,
             5.05432079e-04, 6.98442513e-04, 2.09254664e-04, 1.88501159e-04,
             1.94052409e-04, 8.06364769e-05, 8.41256697e-04, 2.93829740e-04,
             1.63241784e-04, 9.82471975e-04, 1.51760920e-04, 7.25031039e-03,
             3.50983144e-04, 1.15065719e-03, 6.30916655e-03, 3.57609778e-03,
             2.23088355e-04, 3.67130298e-04, 1.27942231e-03, 7.86580349e-05,
             3.40563187e-04, 9.62705933e-04, 2.30116930e-04, 4.89595206e-03,
             7.38742092e-05, 2.24681033e-04, 4.25187172e-03, 3.49205919e-04,
             4.84270568e-04, 6.29964998e-05, 8.33525904e-04, 3.25042522e-04,
             1.41819438e-03, 4.66999802e-04, 1.34147384e-04, 2.43368163e-03,
             7.42314383e-04, 3.96344578e-04, 8.55755527e-04, 1.66909874e-03,
             3.56280565e-04, 6.41367573e-04, 1.33493589e-03, 5.62281231e-04,
             4.38060029e-04, 1.25840888e-03, 3.63592640e-04, 4.09116255e-05,
             1.17004864e-04, 3.65237880e-04, 7.79753900e-04, 2.50286044e-04,
             4.95144704e-05, 1.52532698e-03, 5.80310181e-04, 6.86694577e-04,
             4.08832530e-05, 1.17303789e-04, 5.44532761e-03, 2.17165347e-04,
             3.54671967e-03, 1.21751358e-03, 2.19386551e-04, 3.99591845e-05,
             1.92817824e-04, 1.51990494e-03, 9.52329792e-05, 2.07933085e-03,
             4.49196057e-04, 2.32176695e-04, 1.97328452e-04, 1.90497580e-04,
             1.54137122e-03, 2.79319444e-04, 3.55583004e-04, 6.67025161e-04,
             5.37929591e-04, 1.16214513e-04, 2.04684766e-04, 2.00985558e-03,
             4.77121095e-04, 1.43001729e-04, 6.98699616e-04, 4.39056254e-04,
             1.18351460e-03, 4.55539121e-04, 2.41027563e-04, 3.74429306e-04,
             1.28518650e-03, 4.97089313e-05, 3.76588374e-04, 7.92419305e-05,
             3.07888491e-04, 1.67703183e-04, 1.19715267e-04, 5.82329114e-04,
             3.67476488e-03, 2.39737943e-04, 3.00253305e-04, 1.42522750e-03,
             5.74446004e-03, 5.98638842e-04, 9.94725342e-05, 5.11667808e-04,
             3.06425180e-04, 3.56745499e-04, 5.54735190e-04, 1.94834589e-04,
             6.80337253e-04, 4.59827064e-03, 5.04539465e-04, 5.27943484e-04,
             7.84871168e-04, 5.80510299e-04, 2.62445142e-03, 4.80970164e-04,
             3.11612421e-05, 2.41081478e-04, 1.28599629e-03, 7.65955949e-04,
             2.36732376e-04, 3.60864855e-04, 1.52698404e-03, 2.55799619e-04,
             1.44351230e-04, 4.70804749e-03, 1.70068670e-04, 3.35329278e-05,
             1.53403089e-04, 2.21775193e-02, 7.83315845e-05, 3.03817657e-03,
             1.76668444e-04, 8.71487719e-04, 1.23689184e-03, 7.00269229e-05,
             3.25480546e-03, 3.88289016e-04, 7.85870710e-04, 6.48489920e-04,
             3.15491052e-04, 1.22149795e-04, 6.54125470e-05, 1.43746100e-03,
             5.67841344e-04, 2.25909942e-04, 1.06003245e-04, 4.32289904e-04,
             2.45661213e-04, 9.79025717e-05, 8.55812861e-04, 8.86444832e-05,
             8.08648882e-04, 1.99693051e-04, 1.44935132e-03, 6.84804691e-05,
             1.22172874e-04, 1.03707192e-03, 2.62837624e-04, 1.77469978e-04,
             8.05136879e-05, 3.20125728e-05, 9.92525966e-05, 7.60346476e-04,
             5.40740788e-04, 8.80005071e-04, 2.46038107e-04, 7.89300248e-04,
             1.69093953e-03, 1.90125592e-03, 3.05495254e-04, 7.77769694e-03,
             8.83247296e-04, 5.47260337e-04, 1.11473353e-04, 7.52284395e-05,
             7.75661319e-04, 4.05635656e-04, 2.86911236e-04, 5.06639422e-04,
             1.50237558e-03, 1.94514054e-03, 1.61218297e-04, 1.25441016e-04,
             1.54908397e-03, 2.38077925e-03, 1.43390184e-03, 2.17004083e-04,
             1.93641550e-04, 2.29592697e-04, 2.68609985e-03, 6.20115956e-04,
             3.85113381e-05, 4.91162296e-04, 1.32817557e-04, 8.24844697e-04,
             6.77522854e-04, 6.70986599e-04, 3.41747049e-03, 1.79517164e-03,
             2.41998630e-03, 1.74194225e-04, 2.62646633e-03, 2.49535777e-03,
             1.27389998e-04, 2.88204959e-04, 1.35887589e-03, 1.17561940e-04,
             3.98256670e-04, 1.13561971e-03, 4.79865441e-04, 2.87160208e-03,
             5.27064665e-04, 9.13271913e-04, 1.29025691e-04, 5.18187706e-04,
             1.50446594e-03, 2.68277770e-04, 2.97207851e-04, 2.21228693e-04,
             8.70247837e-04, 4.99874004e-04, 8.36461695e-05, 4.38912888e-04,
             1.30146078e-03, 3.54595686e-05, 6.38484256e-04, 2.86855455e-03,
             1.13861950e-03, 2.82211200e-04, 7.42189994e-04, 9.37753764e-04,
             6.06058165e-04, 1.39768724e-03, 1.72652726e-04, 3.45179695e-04,
             3.17267870e-04, 8.21181398e-04, 2.90494470e-04, 2.78704986e-03,
             9.89483669e-05, 3.97867152e-05, 3.17839975e-03, 7.39140960e-05,
             3.11297714e-04, 1.18029602e-04, 1.49727031e-03, 9.32373150e-05,
             6.02206739e-04, 2.18775105e-02, 1.35777605e-04, 1.11070425e-04,
             2.33035957e-04, 2.44134758e-03, 2.13422906e-03, 1.94211636e-04,
             4.23009798e-04, 1.00112788e-03, 6.23553409e-04, 5.43271250e-04,
             4.95019741e-03, 1.36633753e-03, 3.73949704e-04, 3.62116582e-04,
             1.23001949e-03, 4.93682863e-04, 1.55703095e-03, 1.20034223e-04,
             5.06577373e-04, 3.00323474e-04, 6.95395793e-05, 2.31223786e-03,
             2.46237521e-03, 2.91230390e-04, 1.61908800e-04, 2.13369742e-04,
             1.01302017e-03, 3.20179650e-04, 7.50229592e-05, 1.93457250e-04,
             1.07499827e-02, 1.21929776e-03, 1.98667200e-04, 4.32141533e-05,
             6.85359410e-04, 4.37820454e-05, 1.61343414e-04, 2.79039901e-04,
             7.00738281e-04, 1.98924402e-03, 6.42405299e-04, 1.31197611e-03,
             1.04761207e-02, 5.36928140e-03, 2.05728831e-03, 6.73542439e-04,
             3.28243739e-04, 3.30398034e-05, 3.36833269e-04, 1.21125684e-03,
             2.49467697e-03, 2.53728568e-03, 3.22172279e-03, 4.00726043e-04,
             1.02865495e-04, 1.41505315e-03, 4.12442780e-04, 4.54992463e-04,
             6.20846823e-03, 9.16509831e-04, 2.05098244e-04, 2.03037495e-03,
             1.92099978e-04, 2.11120947e-04, 1.00640187e-04, 5.16679429e-04,
             9.11996030e-05, 5.20647503e-04, 2.92261480e-04, 1.33459526e-03,
             3.59861326e-04, 4.32286120e-04, 8.40913330e-04, 2.19447538e-04,
             5.21354144e-04, 1.34750665e-03, 1.33568712e-03, 3.90007510e-04,
             2.00536777e-03, 1.89733901e-03, 3.79739865e-03, 3.75419564e-04,
             1.06911489e-03, 2.52585113e-03, 2.05417600e-04, 9.46892891e-04,
             3.60171864e-04, 1.53573987e-04, 3.41700041e-04, 1.43267025e-04,
             2.20625172e-03, 2.98218895e-03, 5.18341876e-05, 6.44216198e-04,
             9.08498478e-04, 5.05098898e-04, 9.96567906e-05, 3.63851083e-03,
             5.73441444e-04, 2.68241693e-03, 5.51492325e-04, 7.11930392e-04,
             2.60008080e-03, 1.29773282e-03, 9.84572456e-04, 1.45558879e-04,
             1.54208508e-04, 5.89131203e-04, 1.62220921e-03, 6.59886064e-05,
             4.37661627e-04, 1.14545313e-04, 5.23960160e-04, 1.92338135e-03,
             1.95227878e-03, 6.68951892e-04, 4.78718139e-04, 5.80485619e-04,
             1.29596097e-04, 1.71451783e-03, 1.72232132e-04, 3.98013100e-04,
             1.08482223e-03, 3.99181666e-03, 1.25988377e-02, 3.54024320e-04,
             3.15872632e-04, 7.30704283e-04, 1.07844232e-03, 9.55459196e-04,
             4.06526378e-05, 5.17479260e-04, 1.79097173e-04, 5.42399881e-04,
             2.68625934e-03, 5.51257399e-04, 2.40965179e-04, 2.19254565e-04,
             2.53107981e-04, 9.76901152e-04, 7.58429815e-04, 7.09607790e-04,
             1.63618254e-03, 3.42325168e-03, 1.50652952e-04, 7.19388307e-04,
             1.32352195e-03, 4.94101296e-05, 7.73851294e-04, 4.00231802e-05,
             7.63385688e-05, 1.14887909e-04, 9.73806018e-05, 7.52514927e-04,
             3.02035623e-04, 1.56946160e-04, 5.09049394e-04, 1.16036710e-04,
             1.91065506e-03, 4.12572808e-05, 3.36537545e-04, 1.53840485e-03,
             3.98135337e-04, 2.63095688e-04, 2.85411777e-04, 1.90599880e-04,
             5.74480800e-04, 6.84043625e-04, 2.12100334e-03, 1.22691170e-04,
             1.17575563e-03, 3.93969851e-04, 6.43996755e-04, 1.67475897e-04,
             2.35600793e-03, 4.82445396e-03, 1.00002124e-03, 1.42321660e-04,
             4.70067927e-04, 1.57377869e-03, 1.09559158e-03, 7.51297339e-04,
             6.12211414e-04, 8.93370452e-05, 5.24919815e-05, 1.75424060e-03,
             3.75528944e-05, 1.32379588e-04, 2.90347409e-04, 1.29595050e-03,
             6.18817940e-05, 8.03718437e-03, 2.52122059e-04, 1.04768515e-04,
             6.41977822e-05, 3.81848840e-05, 8.65068505e-05, 1.34375689e-04,
             4.44128271e-03, 2.06961005e-04, 2.01896051e-04, 1.63798191e-04,
             4.19303367e-04, 1.43744648e-04, 6.53140887e-05, 3.08282790e-04,
             2.53811758e-03, 2.10349928e-04, 1.02821898e-04, 1.87941303e-04,
             2.18156638e-04, 1.46553692e-04, 7.28555257e-04, 1.66584272e-03,
             3.80202872e-03, 8.55638587e-04, 9.40019745e-05, 2.26433098e-04,
             1.28183354e-04, 1.54665977e-04, 3.48230998e-04, 4.43409313e-04,
             4.28056763e-03, 1.97874615e-03, 9.82739497e-04, 8.25623574e-04,
             1.92055071e-03, 2.86227185e-03, 1.99538469e-03, 1.93706655e-03,
             8.44598166e-04, 3.13871866e-03, 1.33611809e-03, 2.45591276e-04,
             1.37292978e-03, 9.53664014e-04, 6.27337897e-04, 5.38632157e-04,
             3.18714743e-03, 5.25624375e-04, 1.92286808e-03, 1.62739551e-03,
             8.67567724e-05, 6.26091263e-04, 3.65151354e-04, 2.83986883e-04,
             8.93486358e-05, 9.20956809e-05, 1.46665866e-03, 5.58349013e-04,
             1.78683636e-04, 5.02084661e-03, 2.57270574e-03, 4.10788285e-04,
             9.49688256e-05, 1.21567294e-03, 6.96088828e-05, 2.95326259e-04,
             5.60771325e-04, 9.76681593e-04, 2.86626688e-04, 8.83748231e-04,
             3.66802386e-04, 3.45919194e-04, 2.64546921e-04, 5.54996077e-04,
             2.12450250e-04, 1.80235147e-04, 4.37748124e-04, 1.24568498e-04,
             2.99113832e-04, 1.67661288e-04, 1.07449933e-03, 2.63058289e-04,
             1.65628822e-04, 1.49627376e-04, 7.13991612e-05, 2.29093697e-04,
             7.92245904e-04, 8.87540373e-05, 1.97451838e-04, 2.86982686e-04,
             2.55839754e-04, 1.27500971e-04, 4.15523857e-04, 9.10318849e-05,
             3.30112380e-04, 1.73237830e-04, 1.62401484e-04, 1.98280613e-04,
             2.01226736e-04, 1.40036931e-02, 8.90737938e-05, 9.36848446e-05,
             2.43386385e-04, 1.45516102e-03, 5.43178292e-04, 5.66090865e-04,
             6.56481832e-04, 5.13605890e-04, 4.23861376e-04, 1.44793396e-03,
             1.95318673e-04, 4.74969129e-05, 5.00976050e-04, 2.01932446e-04,
             1.90048013e-04, 3.42751550e-03, 2.51341582e-04, 1.32606583e-04,
             4.84845805e-04, 2.32631181e-04, 3.67411383e-04, 1.21836783e-04,
             1.39171840e-04, 1.76577829e-04, 5.31608530e-05, 2.99918174e-04,
             4.49622079e-04, 1.68509097e-04, 3.99499622e-05, 6.46771950e-05,
             1.52547524e-04, 2.50590292e-05, 1.33419831e-04, 2.01397808e-04,
             1.48847175e-04, 1.33264053e-04, 1.79059280e-04, 2.17893291e-02]],
           dtype=float32)>]




```python
len(extracted_features)
```




    26




```python
len(features_list)
```




    26



这尤其适用于诸如[神经风格迁移](https://tensorflow.google.cn/tutorials/generative/style_transfer?hl=zh-cn)之类的任务。

## 使用自定义层扩展 API
tf.keras 包含了各种内置层，例如：

- 卷积层：Conv1D、Conv2D、Conv3D、Conv2DTranspose
- 池化层：MaxPooling1D、MaxPooling2D、MaxPooling3D、AveragePooling1D
- RNN 层：GRU、LSTM、ConvLSTM2D
- BatchNormalization、Dropout、Embedding 等

但是，如果找不到所需内容，可以通过创建您自己的层来方便地扩展 API。所有层都会子类化 Layer 类并实现下列方法：

- call 方法，用于指定由层完成的计算。
- build 方法，用于创建层的权重（这只是一种样式约定，因为您也可以在 __init__ 中创建权重）。

要详细了解从头开始创建层的详细信息，请阅读自定义层和模型指南。

以下是 tf.keras.layers.Dense 的基本实现：


```python
class CustomDense(layers.Layer):
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)

model = keras.Model(inputs, outputs)
```

为了在您的自定义层中支持序列化，请定义一个get_config方法，该方法返回该层实例的构造函数参数：


```python
class CustomDense(layers.Layer):
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {"units": self.units}


inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)

model = keras.Model(inputs, outputs)
config = model.get_config()

new_model = keras.Model.from_config(config, custom_objects={"CustomDense": CustomDense})
```

您也可以选择实现 from_config(cls, config) 类方法，该方法用于在给定其配置字典的情况下重新创建层实例。from_config 的默认实现如下：


```python
def from_config(cls, config):   return cls(**config)
```

## 何时使用函数式 API
什么时候应该使用 Keras 函数式 API 来创建新的模型，或者什么时候应该直接对 Model 类进行子类化呢？通常来说，函数式 API 更高级、更易用且更安全，并且具有许多子类化模型所不支持的功能。

但是，当构建不容易表示为有向无环的层计算图的模型时，模型子类化会提供更大的灵活性。例如，您无法使用函数式 API 来实现 Tree-RNN，而必须直接子类化 Model 类。

要深入了解函数式 API 和模型子类化之间的区别，请阅读 [TensorFlow 2.0 符号式 API 和命令式 API 介绍](https://blog.tensorflow.org/2019/01/what-are-symbolic-and-imperative-apis.html?hl=zh-cn&_gl=1*14zbuvm*_ga*MTU5NjY0OTI3MC4xNjEyMjczODQ0*_ga_W0YLR4190T*MTY4MTY1ODQ4OS4xNy4xLjE2ODE2NTk4OTEuMC4wLjA.)。

### 函数式 API 的优势
下列属性对于序列模型（也是数据结构）同样适用，但对于子类化模型（是 Python 字节码而非数据结构）则不适用。
#### 更加简洁
没有 super(MyClass, self).__init__(...)，没有 def call(self, ...): 等内容。

对比：


```python
# 函数式API：
inputs = keras.Input(shape=(32,)) 
x = layers.Dense(64, activation='relu')(inputs) 
outputs = layers.Dense(10)(x) 
mlp = keras.Model(inputs, outputs)
```


```python
#下面是子类化版本：
class MLP(keras.Model):    
    def __init__(self, **kwargs):     
        super(MLP, self).__init__(**kwargs)     
        self.dense_1 = layers.Dense(64, activation='relu')     
        self.dense_2 = layers.Dense(10)    
    def call(self, inputs):     
        x = self.dense_1(inputs)     
        return self.dense_2(x)  
# Instantiate the model. 
mlp = MLP() 
# Necessary to create the model's state. 
# The model doesn't have a state until it's called at least once. 
_ = mlp(tf.zeros((1, 32)))
```

#### 定义连接计算图时进行模型验证
在函数式 API 中，输入规范（形状和 dtype）是预先创建的（使用 Input）。每次调用层时，该层都会检查传递给它的规范是否符合其假设，如不符合，它将引发有用的错误消息。

这样可以保证能够使用函数式 API 构建的任何模型都可以运行。所有调试（除与收敛有关的调试外）均在模型构造的过程中静态发生，而不是在执行时发生。这类似于编译器中的类型检查。

#### 函数式模型可绘制且可检查
您可以将模型绘制为计算图，并且可以轻松访问该计算图中的中间节点。例如，要提取和重用中间层的激活（如前面的示例所示），请运行以下代码：


```python
features_list = [layer.output for layer in vgg19.layers] 
feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)
```

#### 函数式模型可以序列化或克隆
因为函数式模型是数据结构而非一段代码，所以它可以安全地序列化，并且可以保存为单个文件，从而使您可以重新创建完全相同的模型，而无需访问任何原始代码。请参阅序列化和保存指南。

要序列化子类化模型，实现器必须在模型级别指定 get_config() 和 from_config() 方法。

### 函数式 API 的劣势
#### 不支持动态架构
函数式 API 将模型视为层的 DAG。对于大多数深度学习架构来说确实如此，但并非所有（例如，递归网络或 Tree RNN 就不遵循此假设，无法在函数式 API 中实现）

## 混搭API和子类化模型
在函数式 API 或模型子类化之间进行选择并非是让您作出二选一的决定而将您限制在某一类模型中。tf.keras API 中的所有模型都可以彼此交互，无论它们是 Sequential 模型、函数式模型，还是从头开始编写的子类化模型。

您始终可以将函数式模型或 Sequential 模型用作子类化模型或层的一部分：


```python
units = 32
timesteps = 10
input_dim = 5

# Define a Functional model
inputs = keras.Input((None, units))
x = layers.GlobalAveragePooling1D()(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)


class CustomRNN(layers.Layer):
    def __init__(self):
        super(CustomRNN, self).__init__()
        self.units = units
        self.projection_1 = layers.Dense(units=units, activation="tanh")
        self.projection_2 = layers.Dense(units=units, activation="tanh")
        # Our previously-defined Functional model
        self.classifier = model

    def call(self, inputs):
        outputs = []
        state = tf.zeros(shape=(inputs.shape[0], self.units))
        for t in range(inputs.shape[1]):
            x = inputs[:, t, :]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
        features = tf.stack(outputs, axis=1)
        print(features.shape)
        return self.classifier(features)


rnn_model = CustomRNN()
_ = rnn_model(tf.zeros((1, timesteps, input_dim)))
```

    (1, 10, 32)


您可以在函数式 API 中使用任何子类化层或模型，前提是它实现了遵循以下模式之一的 call 方法：

- call(self, inputs, **kwargs) - 其中 inputs 是张量或张量的嵌套结构（例如张量列表），**kwargs 是非张量参数（非输入）。
- call(self, inputs, training=None, **kwargs) - 其中 training 是指示该层是否应在训练模式和推断模式下运行的布尔值。
- call(self, inputs, mask=None, **kwargs) - 其中 mask 是一个布尔掩码张量（对 RNN 等十分有用）。
- call(self, inputs, training=None, mask=None, **kwargs) - 当然，您可以同时具有掩码和训练特有的行为。

此外，如果您在自定义层或模型上实现了 get_config 方法，则您创建的函数式模型将仍可序列化和克隆。

下面是一个从头开始编写、用于函数式模型的自定义 RNN 的简单示例：


```python
units = 32
timesteps = 10
input_dim = 5
batch_size = 16


class CustomRNN(layers.Layer):
    def __init__(self):
        super(CustomRNN, self).__init__()
        self.units = units
        self.projection_1 = layers.Dense(units=units, activation="tanh")
        self.projection_2 = layers.Dense(units=units, activation="tanh")
        self.classifier = layers.Dense(1)

    def call(self, inputs):
        outputs = []
        state = tf.zeros(shape=(inputs.shape[0], self.units))
        for t in range(inputs.shape[1]):
            x = inputs[:, t, :]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
        features = tf.stack(outputs, axis=1)
        return self.classifier(features)


# Note that you specify a static batch size for the inputs with the `batch_shape`
# arg, because the inner computation of `CustomRNN` requires a static batch size
# (when you create the `state` zeros tensor).
inputs = keras.Input(batch_shape=(batch_size, timesteps, input_dim))
x = layers.Conv1D(32, 3)(inputs)
outputs = CustomRNN()(x)

model = keras.Model(inputs, outputs)

rnn_model = CustomRNN()
_ = rnn_model(tf.zeros((1, 10, 5)))
```

# 通过子类化创建新的层和模型



```python
import tensorflow as tf
from tensorflow import keras
```

## Layer 类：状态（权重）和部分计算的组合
Layer类是Keras的核心抽象之一，其封装了状态（层的“权重”）和从输入到输出的转换（“call”，即层的前向传递）。

下面是一个密集连接的层。它具有一个状态：变量 w 和 b。


```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

您可以在某些张量输入上通过调用来使用层，这一点很像 Python 函数。


```python
x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
```

    tf.Tensor(
    [[-0.10001414  0.08780711 -0.08834612 -0.03358569]
     [-0.10001414  0.08780711 -0.08834612 -0.03358569]], shape=(2, 4), dtype=float32)


请注意，权重 w 和 b 在被设置为层特性后会由层自动跟踪：


```python
assert linear_layer.weights == [linear_layer.w, linear_layer.b]
```

请注意，您还可以使用一种更加快捷的方式为层添加权重：add_weight() 方法：


```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
```

    tf.Tensor(
    [[-0.00415667 -0.00145001  0.06350404  0.08343165]
     [-0.00415667 -0.00145001  0.06350404  0.08343165]], shape=(2, 4), dtype=float32)


## Layer可以具有不可训练权重
除了可训练权重外，您还可以向层添加不可训练权重。训练层时，不必在反向传播期间考虑此类权重。

以下是添加和使用不可训练权重的方式：


```python
class ComputeSum(keras.layers.Layer):
    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total


x = tf.ones((2, 2))
my_sum = ComputeSum(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())
```

    [2. 2.]
    [4. 4.]


它是 layer.weights 的一部分，但被归类为不可训练权重：





```python
print("weights:", len(my_sum.weights))
print("non-trainable weights:", len(my_sum.non_trainable_weights))

# It's not included in the trainable weights:
print("trainable_weights:", my_sum.trainable_weights)
```

    weights: 1
    non-trainable weights: 1
    trainable_weights: []


## 最佳做法：将权重创建推迟到得知输入的形状之后
上面的 Linear 层接受了一个 input_dim 参数，用于计算 __init__() 中权重 w 和 b 的形状：


```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

在许多情况下，您可能事先不知道输入的大小，并希望在得知该值时（对层进行实例化后的某个时间）再延迟创建权重。

在 Keras API 中，我们建议您在层的 build(self, inputs_shape) 方法中创建层权重。如下所示：


```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

层的 __call__() 方法将在首次调用时自动运行构建。现在，您有了一个延迟并因此更易使用的层：


```python
# 在实例化时，我们不知道会调用什么输入
linear_layer = Linear(32)

# 层的权重是在第一次调用该层时动态创建的
y = linear_layer(x)
```

如上所示，单独实现build（）很好地将只创建一次权重与在每次调用中使用权重区分开来。然而，对于一些高级自定义层，将状态创建和计算分离可能变得不切实际。允许层实现者将权重创建推迟到第一个\__call__（），但需要注意以后的调用使用相同的权重。此外，由于\__call__（）很可能是第一次在tf.function中执行，所以在\__call___（）中发生的任何变量创建都应该封装在atf.init_scope中。

## Layer是可以递归组合的
如果将层实例作为另一个层的属性，则外部层将开始跟踪内部层创建的权重。

我们建议在\__init__() 方法中创建此类子层，并让 \__call__() 中第一个来触发构建层的权重。


```python
class MLPBlock(keras.layers.Layer):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(1)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)


mlp = MLPBlock()
y = mlp(tf.ones(shape=(3, 64)))  # The first call to the `mlp` will create the weights
print("weights:", len(mlp.weights))
print("trainable weights:", len(mlp.trainable_weights))
```

    weights: 6
    trainable weights: 6


## add_loss方法
编写层的回调方法时，可以训练流中创建稍后想用到的损失张量，通过调用 self.add_loss(value) 来实现：


```python
# A layer that creates an activity regularization loss
class ActivityRegularizationLayer(keras.layers.Layer):
    def __init__(self, rate=1e-2):
        super(ActivityRegularizationLayer, self).__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs
```

这些损失（包括由任何内部层创建的损失）可通过 layer.losses 取到，此属性会在每个 __call__() 开始时重置到顶层，因此 layer.losses 始终包含在上一次前向传递过程中创建的损失值。


```python
class OuterLayer(keras.layers.Layer):
    def __init__(self):
        super(OuterLayer, self).__init__()
        self.activity_reg = ActivityRegularizationLayer(1e-2)

    def call(self, inputs):
        return self.activity_reg(inputs)


layer = OuterLayer()
assert len(layer.losses) == 0  # No losses yet since the layer has never been called
```


```python
layer.losses
```




    []




```python
_ = layer(tf.zeros(1, 1))
_ 
```




    <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.], dtype=float32)>




```python
layer.losses
```




    [<tf.Tensor: shape=(), dtype=float32, numpy=0.0>]




```python
_ = layer(tf.zeros(1, 1))
_
```




    <tf.Tensor: shape=(1,), dtype=float32, numpy=array([0.], dtype=float32)>




```python
layer.losses
```




    [<tf.Tensor: shape=(), dtype=float32, numpy=0.0>]




```python
_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # We created one loss value

# `layer.losses` gets reset at the start of each __call__
_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # This is the loss created during the call above
```

loss属性还包括给任何内部层的权重创建的正则化损失


```python
class OuterLayerWithKernelRegularizer(keras.layers.Layer):
    def __init__(self):
        super(OuterLayerWithKernelRegularizer, self).__init__()
        self.dense = keras.layers.Dense(
            32, kernel_regularizer=tf.keras.regularizers.l2(1e-3)
        )

    def call(self, inputs):
        return self.dense(inputs)


layer = OuterLayerWithKernelRegularizer()
_ = layer(tf.zeros((1, 1)))

# This is `1e-3 * sum(layer.dense.kernel ** 2)`,
# created by the `kernel_regularizer` above.
print(_)
print(layer.losses)
```

    tf.Tensor(
    [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
      0. 0. 0. 0. 0. 0. 0. 0.]], shape=(1, 32), dtype=float32)
    [<tf.Tensor: shape=(), dtype=float32, numpy=0.0021133565>]


在编写训练循环时要考虑这些损失，如下所示:


```python
# Instantiate an optimizer. 实例化优化器
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Iterate over the batches of a dataset. # 遍历训练数据
for x_batch_train, y_batch_train in train_dataset:
  with tf.GradientTape() as tape:
    logits = layer(x_batch_train)  # Logits for this minibatch 
    # Loss value for this minibatch
    loss_value = loss_fn(y_batch_train, logits)  # 损失值
    # Add extra losses created during this forward pass:
    loss_value += sum(model.losses)

  grads = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(grads, model.trainable_weights))
```

有关编写训练循环的详细指南，请参阅[从头开始编写训练循环指南](https://tensorflow.google.cn/guide/keras/writing_a_training_loop_from_scratch/?hl=zh-cn)。

这些损失还可以使用 fit()工作（它们会自动求和并添加到主损失中，如果有）：


```python
import numpy as np

inputs = keras.Input(shape=(3,))
outputs = ActivityRegularizationLayer()(inputs)
model = keras.Model(inputs, outputs)

# If there is a loss passed in `compile`, the regularization
# losses get added to it
model.compile(optimizer="adam", loss="mse")   # 自定义损失加入到mse主损失
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

# It's also possible not to pass any loss in `compile`,
# since the model already has a loss to minimize, via the `add_loss`
# call during the forward pass!
model.compile(optimizer="adam")         # 只有自定义损失
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))
```

    1/1 [==============================] - 0s 88ms/step - loss: 0.1850
    1/1 [==============================] - 0s 35ms/step - loss: 0.0382





    <keras.callbacks.History at 0x7ff558402c70>



## add_metric() 方法
与add_loss()类似，层也有一个add_metric()方法，用于在训练期间跟踪数量的移动平均。

请思考下面的 "logistic endpoint" 层。它将预测和目标作为输入，计算通过 add_loss() 跟踪的损失，并计算通过 add_metric() 跟踪的准确率标量。


```python
class LogisticEndpoint(keras.layers.Layer):
    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_fn = keras.metrics.BinaryAccuracy()

    def call(self, targets, logits, sample_weights=None):
        # Compute the training-time loss value and add it    计算训练时损失值并将其加入到层中
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # Log accuracy as a metric and add it   记录准确性作为度量将其添加到层中
        # to the layer using `self.add_metric()`.
        acc = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc, name="accuracy")

        # Return the inference-time prediction tensor (for `.predict()`).
        return tf.nn.softmax(logits)
```

可通过 layer.metrics 跟踪的度量指标：


```python
layer = LogisticEndpoint()

targets = tf.ones((2, 2))
logits = tf.ones((2, 2))
y = layer(targets, logits)

print("layer.metrics:", layer.metrics)
print("current accuracy value:", float(layer.metrics[0].result()))
```

    layer.metrics: [<keras.metrics.metrics.BinaryAccuracy object at 0x7ff558318e20>]
    current accuracy value: 1.0


和 add_loss() 一样，这些指标也是通过 fit() 跟踪的：


```python
inputs = keras.Input(shape=(3,), name="inputs")
targets = keras.Input(shape=(10,), name="targets")
logits = keras.layers.Dense(10)(inputs) # 函数式，默认调用call方法
predictions = LogisticEndpoint(name="predictions")(logits, targets) # 自定义层，调用call，传入call所需参数

model = keras.Model(inputs=[inputs, targets], outputs=predictions)
model.compile(optimizer="adam")

data = {
    "inputs": np.random.random((3, 3)),
    "targets": np.random.random((3, 10)),
}
model.fit(data)
```

    1/1 [==============================] - 0s 413ms/step - loss: 1.0551 - binary_accuracy: 0.0000e+00





    <keras.callbacks.History at 0x7ff55838f880>



## 可选择在层上启用序列化

如果需要将自定义层作为函数式模型的一部分进行序列化（指的是可以保存用于外部加载），实现get_config() 方法即可：


```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        return {"units": self.units}


# Now you can recreate the layer from its config: 可以通过config重建层
layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)
```

    {'units': 64}


请注意，基础 Layer 类的 __init__() 方法会接受一些关键字参数，尤其是 name 和 dtype。最好将这些参数通过 __init__() 传递给父类，并将其包含在层配置中：


```python
class Linear(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({"units": self.units})
        return config


layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)
```

    {'name': 'linear_9', 'trainable': True, 'dtype': 'float32', 'units': 64}


如果根据层的配置对层进行反序列化时需要更大的灵活性，还可以重写 from_config() 类方法。下面是 from_config() 的基础实现：





```python
def from_config(cls, config):
  return cls(**config)
```

## call() 方法中的特权 training 参数
某些层，尤其是 BatchNormalization 层和 Dropout 层，在训练和推断期间具有不同的行为。对于此类层，标准做法是在 call() 方法中公开 training（布尔）参数。

通过在 call() 中公开此参数，可以启用内置的训练和评估循环（例如 fit()）以在训练和推断中正确使用层。


```python
class CustomDropout(keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs
```

## call() 方法中的特权 mask 参数
call() 支持的另一个特权参数是 mask 参数。

它会出现在所有 Keras RNN 层中。掩码是布尔张量（在输入中每个时间步骤对应一个布尔值），用于在处理时间序列数据时跳过某些输入时间步骤。

当先前的层生成掩码时，Keras 会自动将正确的 mask 参数传递给 __call__()（针对支持它的层）。掩码生成层是配置了 mask_zero=True 的 Embedding 层和 Masking 层。

要详细了解遮盖以及如何编写启用遮盖的层，请查看了解[填充和遮盖指南](https://tensorflow.google.cn/guide/keras/masking_and_padding/?hl=zh-cn)。

## Model 类
通常，您会使用 Layer 类来定义内部计算块，并使用 Model 类来定义外部模型，即您将训练的对象。

例如，在 ResNet50 模型中，您会有几个子类化 Layer 的 ResNet 块，以及一个包含整个 ResNet50 网络的 Model。

Model 类具有与 Layer 相同的 API，但有如下区别：

它会公开内置训练、评估和预测循环（model.fit()、model.evaluate()、model.predict()）。
它会通过 model.layers 属性公开其内部层的列表。
它会公开保存和序列化 API（save()、save_weights()…）
实际上，Layer 类对应于我们在文献中所称的“层”（如“卷积层”或“循环层”）或“块”（如“ResNet 块”或“Inception 块”）。

同时，Model 类对应于文献中所称的“模型”（如“深度学习模型”）或“网络”（如“深度神经网络”）。

因此，如果您想知道“我应该用 Layer 类还是 Model 类？”，请问自己：我是否需要在它上面调用 fit()？我是否需要在它上面调用 save()？如果是，则使用 Model。如果不是（要么因为您的类只是更大系统中的一个块，要么因为您正在自己编写训练和保存代码），则使用 Layer。

例如，我们可以使用上面的 mini-resnet 示例，用它来构建一个 Model，该模型可以通过 fit() 进行训练，并通过 save_weights() 进行保存：


```python
class ResNet(tf.keras.Model):

    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        self.block_1 = ResNetBlock()
        self.block_2 = ResNetBlock()
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = Dense(num_classes)

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.global_pool(x)
        return self.classifier(x)


resnet = ResNet()
dataset = ...
resnet.fit(dataset, epochs=10)
resnet.save(filepath)
```

## 汇总：端到端示例
到目前为止，您已学习以下内容：

- Layer 封装了状态（在 __init__() 或 build() 中创建）和一些计算（在 call() 中定义）。
- 层可以递归嵌套以创建新的更大的计算块。
- 层可以通过 add_loss() 和 add_metric() 创建并跟踪损失（通常是正则化损失）以及指标。
- 您要训练的外部容器是 Model。Model 就像 Layer，但是添加了训练和序列化实用工具。
- 让我们将这些内容全部汇总到一个端到端示例：我们将实现一个变分自动编码器 (VAE)，并用 MNIST 数字对其进行训练。

我们的 VAE 将是 Model 的一个子类，它是作为子类化 Layer 的嵌套组合层进行构建的。它将具有正则化损失（KL 散度）。


```python
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, original_dim, intermediate_dim=64, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_output = layers.Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        original_dim,
        intermediate_dim=64,
        latent_dim=32,
        name="autoencoder",
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed
```

让我们在 MNIST 上编写一个简单的训练循环：


```python
original_dim = 784
vae = VariationalAutoEncoder(original_dim, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

loss_metric = tf.keras.metrics.Mean()

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

epochs = 2

# Iterate over epochs.
for epoch in range(epochs):
    print("Start of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vae(x_batch_train)
            # Compute reconstruction loss
            loss = mse_loss_fn(x_batch_train, reconstructed)
            loss += sum(vae.losses)  # Add KLD regularization loss

        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        loss_metric(loss)

        if step % 100 == 0:
            print("step %d: mean loss = %.4f" % (step, loss_metric.result()))
```

    Start of epoch 0
    step 0: mean loss = 0.3591
    step 100: mean loss = 0.1266
    step 200: mean loss = 0.0997
    step 300: mean loss = 0.0895
    step 400: mean loss = 0.0845
    step 500: mean loss = 0.0811
    step 600: mean loss = 0.0789
    step 700: mean loss = 0.0773
    step 800: mean loss = 0.0761
    step 900: mean loss = 0.0751
    Start of epoch 1
    step 0: mean loss = 0.0748
    step 100: mean loss = 0.0741
    step 200: mean loss = 0.0736
    step 300: mean loss = 0.0731
    step 400: mean loss = 0.0728
    step 500: mean loss = 0.0724
    step 600: mean loss = 0.0721
    step 700: mean loss = 0.0718
    step 800: mean loss = 0.0715
    step 900: mean loss = 0.0713


请注意，由于 VAE 是 Model 的子类，它具有内置的训练循环。因此，您也可以用以下方式训练它：


```python
vae = VariationalAutoEncoder(784, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=2, batch_size=64)
```

    Epoch 1/2
    WARNING:tensorflow:5 out of the last 17 calls to <function Model.make_train_function.<locals>.train_function at 0x7ff557a210d0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.


    WARNING:tensorflow:5 out of the last 17 calls to <function Model.make_train_function.<locals>.train_function at 0x7ff557a210d0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.


    938/938 [==============================] - 3s 2ms/step - loss: 0.0748
    Epoch 2/2
    938/938 [==============================] - 2s 2ms/step - loss: 0.0676





    <keras.callbacks.History at 0x7ff558d91c10>



## 超越面向对象的开发：函数式 API
这个示例对您来说是否包含了太多面向对象的开发？您也可以使用函数式 API 来构建模型。重要的是，选择其中一种样式并不妨碍您利用以另一种样式编写的组件：您随时可以搭配使用。


```python
original_dim = 784
intermediate_dim = 64
latent_dim = 32

# Define encoder model.
original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
x = layers.Dense(intermediate_dim, activation="relu")(original_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()((z_mean, z_log_var))
encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")

# Define decoder model.
latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
x = layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
outputs = layers.Dense(original_dim, activation="sigmoid")(x)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

# Define VAE model.
outputs = decoder(z)
vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")

# Add KL divergence regularization loss.
kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)

# Train.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=3, batch_size=64)
```
