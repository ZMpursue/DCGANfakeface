import paddle.fluid as fluid

conv_initializer=fluid.initializer.NormalInitializer(loc=0.0, scale=0.02)
bn_initializer=fluid.initializer.NormalInitializer(loc=1.0, scale=0.02)