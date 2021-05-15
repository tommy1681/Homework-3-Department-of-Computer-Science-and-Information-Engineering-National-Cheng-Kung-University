
# You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()


def output(path, data):
    

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return

def getData(path1,path2):
    
    dateFormatter = "%Y-%m-%d %H:%M:%S"
    df_consumption = pd.read_csv(path1,encoding='utf-8')
    df_generation = pd.read_csv(path2,encoding='utf-8')
    h = len(df_consumption)
    x = []
    for i in range(0,h):
        x.append([df_consumption["consumption"][i],df_generation["generation"][i]])
        x_n = np.array([x])
        last_date = datetime.datetime.strptime(df_consumption["time"][i], dateFormatter)
    print(x_n.shape)
    print(last_date)
    
    
    return x_n,last_date

    
 

def test_model(test_x):
    consumption_model = tf.keras.models.load_model('predict_consumption.h5')
    predict_consumption = consumption_model.predict(test_x)
    generation_model = tf.keras.models.load_model('predict_generation.h5')
    predict_generation = generation_model.predict(test_x)
    print(predict_consumption)
    print(predict_generation)
    return predict_consumption,predict_generation

def rule(predict_consumption,predict_generation,last_date):
    
    
    ans = []
    for i in range(0,len(predict_consumption[0])):
        last_date = last_date + datetime.timedelta(hours=1)
        if predict_consumption[0][i] - predict_generation[0][i] > 1:
            #price = predict_consumption[0][i] - predict_generation[0][i]-0.5
            ans.append([str(last_date),"buy",2.5,1])
            
            
        elif predict_consumption[0][i] - predict_generation[0][i] < -1:
            #price = predict_consumption[0][i] - predict_generation[0][i]+0.5
            ans.append([str(last_date),"sell",2.3,1])
            
            
        else:
            
            print("0")
    
    return ans
if __name__ == "__main__":
    args = config()
    try:
        import pandas as pd
        import numpy as np 
        import tensorflow as tf
        import datetime
        

        test_x,last_date = getData(config().consumption,config().generation)
        predict_consumption,predict_generation = test_model(test_x)
        data = rule(predict_consumption,predict_generation,last_date)
    except:
        data = []
    output(args.output, data)