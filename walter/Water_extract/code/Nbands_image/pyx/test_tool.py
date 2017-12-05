import pyximport
pyximport.install()

#import test15
import tool_set
import numpy as np
from datetime import datetime
if __name__=="__main__":
    # path="/home/wu/Water_extract/data/0_1.tif"
    # img=tool_set.Multiband2Array(path)
    # print(img.shape)
    # pass

    dir_name="/home/wu/Water_extract/data/data/"

    start=datetime.now()

    tool_set.create_pickle_train(dir_name,10,4)
    '''#
    data0=tool_set.read_and_decode(dir_name+'train_data.pkl',10,4)
    print(data0.shape)

    data1 = tool_set.read_and_decode(dir_name + 'train_data_1.pkl', 10, 4)
    print(data1.shape)

    data=np.vstack((data0,data1))
    print(data.shape)
    '''
    print(datetime.now()-start)

