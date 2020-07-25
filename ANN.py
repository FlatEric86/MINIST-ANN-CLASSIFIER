import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import torch 
import torch.nn as nn
import random as rand

plt.style.use('ggplot')


### path of test data
data_path_test = './data/mnist_test.csv'
### path of traing data
data_path_train = './data/mnist_train.csv'

### test data as Data Frame
test  = pd.read_csv(data_path_test)
### training data as Data Frame
train = pd.read_csv(data_path_train)



def test_model():
    '''
    Function to cumpute the model accuracy 
    '''
    model_state_i = net(TEST).to(device)

    count = 0
    with torch.no_grad():
        for i in range(len(TEST)):
          
            real_class      = torch.argmax(label_test[i]).to(device)
            predicted_class = torch.argmax(model_state_i[i]).to(device)
            if predicted_class == real_class:
                count += 1
                           
    return round(count/len(TEST), 3)




### to fetch row data of the dataframes to list
### each list represents an image of a digit as row vector
### Its easer to handle them like so to using some cool options like
### list comprehensions and so on
### In addition we can use some filter functions for preparing
### the data
def get_im_vectors(data_frame):

    VECS = []    
    for i in tqdm(range(len(data_frame))):

        # fetch data from data frame
        vec = list(data_frame.loc[i])[1:]   
        
        # normalize data (we only want to have values of 0 or 1)
        vec = normalize_im(vec)
        
        VECS.append(vec)
        
        # if i == 100:
            # break

        
    return VECS
        
   
### function to set the pixel values to a value in [0, 1] 
### The Criterion is defined so that the value goes 0 if
### the origin pixelvalue is smaller than the 9'th fraction
### of the maximumvalue of all pixel values.
### Otherwise, the value gets to be1
def normalize_im(vec):   
    max_val = max(vec)
    #return [1 if val >= 0.9*max_val else 0 for val in vec]
    return [val/255 for val in vec]
   
        
    
### check if CUDA is possible and store value
use_cuda = torch.cuda.is_available()

#use_cuda = False

### define device in dependency if CUDA is available or not. 
device   = torch.device('cuda:0' if use_cuda else 'cpu')



### define number of CPU cores if device gots defined as CPU
if use_cuda == False:
    torch.set_num_threads(14)   # let one core there for os




    
    
### Test data splittet into the input data (normalized pixel values as vector)
### and the associated labels             
TEST        = torch.tensor(get_im_vectors(test)).float().to(device)
label_test  = torch.tensor([[1 if i == val else 0 for i in range(10)] for val in list(test['label'])]).float().to(device)

### The Same as above with the training data
TRAIN       = torch.tensor(get_im_vectors(train)).float().to(device)
label_train = torch.tensor([[1 if i == val else 0 for i in range(10)] for val in list(train['label'])]).float().to(device)     







class model(nn.Module):
    '''
    This class represents the neuronal network
    '''
    def __init__(self):
        '''
        The class inherits from based parent class nn.Module 
        their respected constructor.
        In addition, there are 3 further attributes which
        represents the layer of the neuronal network.
        '''
        super(model, self).__init__()
        
        # first layer ist the inputlayer which is only to distributes
        # the input values to the first hidden layer (fully connected scheme)
        self.lin1 = nn.Linear(int(28*28), int(28*28))
        
        # second layer (first hidden layer)      
        self.lin2 = nn.Linear(int(28*28), int(28*28))
        
        # third layer (second hidden layer)       
        self.lin6 = nn.Linear(int(28*28), 10)
  
    def forward(self, x):
        '''
        The layer sequence
        '''
        
        x = self.lin1(x)

        x = torch.nn.functional.sigmoid(self.lin2(x))
        x = torch.nn.functional.sigmoid(self.lin6(x))
        
        
        return x #torch.nn.functional.softmax(x)






### number of learning epochs
N_epoch = 2000

### learning rate
lr      = 0.001  
      
### criterion of the loss function
criterion = nn.MSELoss()  


### Define a net object and assign the device.    
net = model().to(device)
    
    
### Load a given model state if it does exist.  
if os.path.isfile('./weights.pt'):
    net.load_state_dict(torch.load('./weights.pt'))      
    
### the model optimizer algorithm    
#optimizer = torch.optim.SGD(net.parameters(), lr=lr)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


loss_function = nn.MSELoss()   
        
        
      
### Array to store the values of the lossfuntion at each epoch
LOSS     = []        
 
### Data Frame for storing the value of model accuracy and the
### associated number of epoch
test_model_res = pd.DataFrame(columns=['epoch', 'accuracy'])        
         

###############################################################################         
########### The Iteration part to optimize the neuronal network ###############
###############################################################################       
for i in range(N_epoch):


    ### set gradient as 0 before optimize the model
    net.zero_grad()
    
    ### model outputs
    outputs = net(TRAIN).to(device)

    ### model loss of the actuall model
    loss = loss_function(outputs, label_train)
    
    
    ### Here we want to get the actuall accuracy of the model
    ### Unfortunatly the function to compute the accuracy needs some time and 
    ### do slow down the Iteration. That's why we define an increment to compute
    ### the accuracy not at each epoch.
    if i % 100 == 0:
        model_acuracy  = test_model()  
        test_model_res = test_model_res.append                                \
        (                                                                     \
            {                                                                 \
            'epoch':i,                                                        \
            'accuracy':model_acuracy                                          \
            },                                                                \
            ignore_index=True                                                 \
        )                                                                     \
    
    print(80*'~')
    print('EPOCH:',i)
    print(float(loss))
    
    with torch.no_grad():
        rand_index = rand.choice([i for i in range(len(label_test))])
        print('Sollwert: ',label_test[rand_index].argmax())
        print('Istwert : ',net(TEST[rand_index]).argmax())
        
    
    ### do backpropagate the model error 
    loss.backward()
    optimizer.step()
   
    LOSS.append(float(loss))
 

### Safe the model state
#torch.save(net.state_dict(), './weights.pt')



### Plotsection to visualize the loss behavior over all epochs as well as
### the evulution of the accuracy over the epochs

fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].plot([i for i in range(len(LOSS))], LOSS, color='green', label='loss')
ax[0].set_title('Evolution of Loss over all Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('Epochs')
ax[0].legend()

 
ax[1].plot                                                                    \
    (                                                                         \
    test_model_res['epoch'],                                                  \
    test_model_res['accuracy'],                                               \
    color='blue', label='accuracy'                                            \
    )                                                                         \
    
ax[1].set_title('Evolution of Accuracy over all Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].legend()

plt.show()
plt.close()



### Here we want to investigate the missclassified numbers by visualizing them. 
with torch.no_grad():
    for i in range(len(TEST)):
        ist_val  = int(net(TEST[i]).argmax())
        soll_val = int(label_test[i].argmax())
        
        if ist_val != soll_val:
            im = TEST[i].cpu().numpy().reshape((28, 28))
            
            print(80*'~')
            print('SOLLWERT : ', soll_val)
            print('ISTWERT  : ', ist_val)
            plt.imshow(im)
            plt.show()
            plt.close()           
        

