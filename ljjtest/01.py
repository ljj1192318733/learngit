import numpy as np
import tensorflow as tf
import dataset

train_batch=50
test_batch=64
para_path='./para/paradata'
iou_path = './data/iou.txt'

class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None, restore_from=None, trainning=False):
        # pre-trained parameters
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('Please download VGG16 parameters from here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM\nOr from my Baidu Cloud: https://pan.baidu.com/s/1Spps1Wy0bvrQHH2IMkRfpg')

        self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.tfy = tf.placeholder(tf.float32, [None, 1])

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])

        # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')

        # detach original VGG fc layers and
        # reconstruct your own fc layers serve for your own purpose
        self.flatten = tf.reshape(pool5, [-1, 7*7*512])
        self.fc6 = tf.layers.dense(self.flatten, 4096, tf.nn.relu, name='fc6')
        self.fc7 = tf.layers.dense(self.fc6, 4096, tf.nn.relu, name='fc7')
        self.out_ = tf.layers.dense(self.fc7, 1, name='out')
        self.out = tf.nn.sigmoid(self.out_)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        if trainning==True:
            self.loss = tf.losses.sigmoid_cross_entropy(self.tfy,self.out_)
            global_step = tf.Variable(0)
            LR = tf.train.exponential_decay(0.0001,global_step,100,0.98,staircase=True)
            self.train_op = tf.train.RMSPropOptimizer(LR).minimize(self.loss,global_step)
            self.sess.run(tf.global_variables_initializer())
        else:
            self.saver.restore(self.sess, restore_from)
        self.saver.restore(self.sess, restore_from)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def train(self, x, y):
        loss, _ ,out,tyf= self.sess.run([self.loss, self.train_op, self.out, self.tfy], {self.tfx: x, self.tfy: y})
        return loss,out,tyf

    def save(self, path=para_path):
        #self.saver = tf.train.Saver()
        self.saver.save(self.sess, path, write_meta_graph=False)

    def pre(self):
        set = dataset.Dataset(test_batch, './data/test_data')
        with tf.Session() as sess:
            _, bad, ys = sess.run(set.batch_next())
            xs = np.array(bad).squeeze()
            pre = self.sess.run(self.out, {self.tfx: xs})
            b_idx = np.random.randint(0, len(xs), test_batch)
            y_ = np.reshape(np.array(ys)[b_idx], [test_batch, 1])
            B = np.array(pre).squeeze()
            A = np.array(y_).squeeze()
            C = np.array([A,B])
            lcc = np.corrcoef(C)/np.std(A)*np.std(B)
            print(A,B,lcc)

def train():
    vgg = Vgg16(vgg16_npy_path='./para/vgg16.npy',
                restore_from = para_path,trainning=True)
    print('Net built')
    set = dataset.Dataset(train_batch,'./data')
    with tf.Session() as sess:
        for i in range(10000):
            src,bad,y = sess.run(set.batch_next())
            # plt.hist(y, bins=50, label='iou')
            # plt.legend()
            # plt.xlabel('iou')
            # plt.show()
            #xs=np.array(bad)+np.array(src)
            xs = np.array(bad).squeeze()
            ys = y
            b_idx = np.random.randint(0, len(xs), train_batch)
            y_ = np.reshape( np.array(ys)[b_idx], [train_batch, 1])
            train_loss,out,tyf = vgg.train(xs[b_idx], y_)
            if i%10==0:
                if i%100==0:
                    print(out,tyf)
                print(i, 'train loss: ', train_loss)
                vgg.save(para_path)      # save learned fc layers

def test():
    vgg = Vgg16(vgg16_npy_path='./para/vgg16.npy',
                restore_from=para_path)
    vgg.pre()

if __name__ == '__main__':
    #train()
    test()