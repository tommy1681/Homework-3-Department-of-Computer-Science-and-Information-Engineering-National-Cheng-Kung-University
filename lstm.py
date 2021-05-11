import pandas as pd
import tensorflow as tf
import numpy as np 

class lstm_model:
    def __init__(self):
        a=1
       
    def build(self,data):

        regressor = tf.keras.Sequential()
        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(tf.keras.layers.LSTM(units=128, return_sequences=True, input_shape=data.shape[-2:]))
  

        # Adding a fourth LSTM layer and some Dropout regularisation
        regressor.add(tf.keras.layers.LSTM(units=50))
        # Adding the output layer
        regressor.add(tf.keras.layers.Dense(units=24))
        
        
        # Compiling the RNN
        regressor.compile(optimizer='Adam', loss='mean_squared_error')
        
        return regressor

    def train(self,paths):
        data = df = pd.read_csv(paths[0],encoding='utf-8')
        x,y= self.data_pre_new(data)
        for i in range(1,len(paths)):
            df = pd.read_csv(paths[i],encoding='utf-8')

            print("資料前處理")
            x_1,y_1= self.data_pre_new(df)
            print(x_1.shape,y_1.shape)
            x = np.vstack([x,x_1])
            y = np.vstack([y,y_1])

        print(y.shape)
        x,y =self.unison_shuffled_copies(x,y)
        x,y,test_x,test_y = self.cut(x,y)
        
        model =self.build(x)
        model.fit(x,y,validation_data = [test_x,test_y] ,epochs=1500,batch_size=128)
        model.save('predict_consumption.h5',save_format='h5')
        print("模型儲存完畢")

        new_model = tf.keras.models.load_model('predict_consumption.h5')
        predict = new_model.predict(test_x)
        print(predict[0])
        print(test_y[0])

        print(predict[1])
        print(test_y[1])
        self.acc(predict,test_y)
    def acc(self,predict,y):
        ac = 0
        for i in range(0,predict.shape[0]):
            for j in range(0,24):
                if predict[i][j]<0:
                    ac+=1

        print(ac)
    def data_pre_new (self,data):
        h = len(data)
        target = []

        for i in range(0,h):
            
            target.append(data["consumption"][i])


        x = []
        y = []
        count = 0
        
        for i in range(0,h,24):
            if i + 192 > h-1 :
                break
            batch = []
            #前3天
            for j in range(0,168):
                a = [data["consumption"][i+j],data["generation"][i+j]]
                batch.append(a)

            x.append(batch)
            ans = []
            #第3到4天
            
            temp = 0
            for k in range(168,192):
                temp = target[i+k]
                ans.append(target[i+k])
                

            y.append(ans)

        x_n = np.array(x)
        y_n = np.array(y)


        return x_n,y_n




    def cut(self,x_n,y_n):
        h = x_n.shape[0]
        spli = int(0.9*h)
        tran_x = x_n[:spli]
        test_x =  x_n[spli:]
        tran_y = y_n[:spli]
        test_y =  y_n[spli:]

        return tran_x,tran_y,test_x,test_y


    def unison_shuffled_copies(self,a, b):
        
        np.random.seed(0)
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
'''
paths = []
for i in range(2,50):
    paths.append("training_data\\target"+str(i)+".csv")
path = "training_data\\target0.csv"
df = pd.read_csv(path,encoding='utf-8')

l = lstm_model()
l.train(paths)
#pre,pre1 = l.data_pre_new(df)
#pre2 = np.vstack([pre,pre])
#print(pre2.shape)'''