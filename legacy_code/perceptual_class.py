import tensorflow as tf


def L2_loss(x, y):
    loss = tf.reduce_mean(tf.square(x - y))

    return loss

class Vgg16_perceptual_loss(tf.keras.Model):
    def __init__(self, class_vgg16):
        super(Vgg16_perceptual_loss, self).__init__(name='Vgg16_perceptual_loss')
        # self.vgg16 = Vgg16_class()
        self.vgg16 = class_vgg16
        self.vgg16_preprocess = tf.keras.applications.vgg16.preprocess_input

    def call(self, x, y):
        x = ((x + 1) / 2) * 255.0
        y = ((y + 1) / 2) * 255.0
        x_vgg, y_vgg = self.vgg16(self.vgg16_preprocess(x)), self.vgg16(self.vgg16_preprocess(y))

        loss = L2_loss(x_vgg, y_vgg)

        return loss

class Inception_v3_feature(tf.keras.Model):
    def __init__(self, class_inception_v3):
        super(Inception_v3_feature, self).__init__(name='Inception_v3_feature')
        # self.inception_v3 = Inception_v3_class()
        self.inception_v3 = class_inception_v3
        self.inception_v3_preprocess = tf.keras.applications.inception_v3.preprocess_input

    def call(self, x):
        x = ((x + 1) / 2) * 255.0

        x_inception = self.inception_v3(self.inception_v3_preprocess(x))

        return x_inception

class Vgg16_class(tf.keras.Model):
    def __init__(self, trainable=False):
        super(Vgg16_class, self).__init__(name='Vgg16_class')
        vgg16_features = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)

        if trainable is False:
            vgg16_features.trainable = False

        self.relu2_2 = tf.keras.Model(inputs=vgg16_features.input, outputs=vgg16_features.get_layer('block2_conv2').output)
        """
        self.relu2_2 = tf.keras.Sequential()

        for i, vgg_layer in enumerate(vgg16_features.layers) :
            if i == 0 :
                continue
            if vgg_layer.name == 'block2_conv2' :
                self.relu2_2.add(vgg_layer)
                break
            else :
                self.relu2_2.add(vgg_layer)
        """

    def call(self, x):

        return self.relu2_2(x)

class Inception_v3_class(tf.keras.Model):
    def __init__(self, trainable=False):
        super(Inception_v3_class, self).__init__(name='Inception_v3_class')
        inception_v3_features = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False)

        if trainable is False:
            inception_v3_features.trainable = False

        self.mixed7 = tf.keras.Model(inputs=inception_v3_features.input, outputs=inception_v3_features.get_layer('mixed7').output)

        """
        self.mixed7 = tf.keras.Sequential()
        for i, inception_v3_layer in enumerate(inception_v3_features.layers) :

            if i == 0 :
                continue

            if inception_v3_layer.name == 'mixed7' :
                self.mixed7.add(inception_v3_layer)
                break

            else :
                self.mixed7.add(inception_v3_layer)
                
        """

    def call(self, x):

        return self.mixed7(x)