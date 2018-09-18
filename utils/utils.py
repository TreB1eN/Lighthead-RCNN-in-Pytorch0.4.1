from torch.nn import Conv2d, Linear
from datetime import datetime

def normal_init(m, mean, stddev):
    if type(m) == Linear or type(m) == Conv2d:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
        
def get_time():
    return (str(datetime.now())[:-10]).replace(' ','-').replace(':','-')
