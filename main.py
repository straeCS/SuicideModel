import pandas as pd
import numpy as np
import os
import time
from load_data import *
from model import *
import warnings
warnings.filterwarnings('ignore')

DEVICE = 'cpu'                  # cpu or cuda
MODEL = 'Transformer'                  # LSTM or Transformer or Linear
TARGETS = ['Seriously considered attempting suicide',
           'Made a plan about how they would attempt suicide',
           'Actually attempted suicide',
           'Suicide attempt resulted in an injury, poisoning, or overdose that had to be treated by a doctor or nurse']        # Target Y, depression data
TIME_STEP = 5                   # How many data points of previous data to use for prediction. Default is 5 (which means 5*2 = 10 years because of every-other-year data)
TEST_START = 2001               # Year to start testing. Default is 2001
NUM_EPOCH = 1000                # training epochs. Default is 1000
LEARNING_RATE = 0.0001          # learning rate of the model. Default is 0.0001
HIDDEN_DIM = 128                # dimension of hidden layers. Default is 128
APPLY_INTERPOLATION = False     # whether use interpolation or not. Default is False


# split dictionary data into training/test tensor data
def split_data(dict_data, test_year):
    list_state, list_depr, list_feat = [], [], []
    for year in range(1999, test_year, 2):
        list_feat += dict_data[str(year) + '_feat']
        list_depr += dict_data[str(year) + '_depr']
        list_state += dict_data[str(year) + '_state']
    
    train_x = np.array(list_feat, dtype=np.float32)         # num_sample * num_feat * num_time_step
    train_y = np.array(list_depr, dtype=np.float32)         # num_sample
    train_state = list_state                                # num_sample

    test_x = np.array(dict_data[str(test_year) + '_feat'], dtype=np.float32)
    test_y = np.array(dict_data[str(test_year) + '_depr'], dtype=np.float32)
    test_state = dict_data[str(test_year) + '_state']

    # return empty if there is no data found (this part of code is never used)
    if len(train_x) <= 0 or len(test_x) <= 0:
        return None, None, [], None, None, []
    
    # reshape the data to meet the requirement of models
    train_x = np.transpose(train_x, (0, 2, 1))              # num_sample * num_time_step * num_feat
    test_x = np.transpose(test_x, (0, 2, 1))

    # turn the data type from numpy to tensor to meet the requirement of models
    train_x = torch.from_numpy(train_x).to(DEVICE)          # num_sample * num_time_step * num_feat
    train_y = torch.from_numpy(train_y).to(DEVICE)          # num_sample
    test_x = torch.from_numpy(test_x).to(DEVICE)
    test_y = torch.from_numpy(test_y).to(DEVICE)

    return train_x, train_y, train_state, test_x, test_y, test_state



if __name__ == '__main__':
    start_time = time.time()
    
    # creat folders to save results
    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists('results/models'):
        os.mkdir('results/models')
    if not os.path.exists('results/shapley'):
        os.mkdir('results/shapley')

    # create a dataframe to save prediction errors
    df_save = pd.DataFrame()
    df_save['Year'] = list(np.arange(TEST_START, 2024, 2))



    # For each Target Y, get a list of prediction
    for cur_target in TARGETS:

        # adjust model hyperparameter based on models and targets to make sure the model would converge
        if MODEL == 'Linear' or (MODEL == 'Transformer' and cur_target == TARGETS[-1]):
            LEARNING_RATE = 0.001
        else:
            LEARNING_RATE = 0.0001

        # load data
        dict_data = load_data(time_step=TIME_STEP, target=cur_target, apply_interpolation=APPLY_INTERPOLATION)

        # For each test year, start model training and test
        save_loss, save_mae, save_per = [], [], []
        for year in range(TEST_START, 2024, 2):
            cur_time = time.time()

            # split data into training and test
            if year == 2023:
                train_x, train_y, train_state, test_x, test_y, test_state = split_data(dict_data, year-2)
                train_x = torch.concatenate([train_x, test_x], dim=0)
                train_y = torch.concatenate([train_y, test_y], dim=0)
            else:
                train_x, train_y, train_state, test_x, test_y, test_state = split_data(dict_data, year)

            # in case there is no data loaded, put -1 to saved result for current year (this part of code is never used)
            if len(test_state) <= 0:
                save_loss.append(-1)
                save_mae.append(-1)
                save_per.append(-1)
                continue
            
            # create model, loss function, and optimizer
            if MODEL == 'Transformer':
                model = TransformerModel(train_x.shape[-1], HIDDEN_DIM, 1, 3).to(DEVICE)
            elif MODEL == 'LSTM':
                model = LSTMModel(train_x.shape[-1], HIDDEN_DIM, 1, 3).to(DEVICE)
            elif MODEL == 'Linear':
                model = LinearRegression(train_x.shape[-1], 1, TIME_STEP).to(DEVICE)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

            # start model training with the training data and save model after well-trained
            model.train()
            if MODEL != 'MovingAverage':
                for epoch in range(NUM_EPOCH):
                    optimizer.zero_grad()
                    pred = model(train_x)
                    loss = criterion(pred, train_y)
                    loss.backward()
                    optimizer.step()
                torch.save(model, 'results/models/' + MODEL + '_' + str(year) + '_' + cur_target[:4] + '.model')

            # start evaluation with the test data
            model.eval()
            pred = model(test_x)
            loss = criterion(pred, test_y)
            save_loss.append(loss.item())                                                       # mean squared error
            save_mae.append(np.mean((torch.abs(pred-test_y)).cpu().detach().numpy()))           # mean absolute error
            save_per.append(np.mean((torch.abs(pred-test_y)/test_y).cpu().detach().numpy()))    # mean error percentage

            print(year, 'Train:', len(train_x), 'Test:', len(test_x), 'MAE Error:', save_mae[-1], 'Total Time Cost:', time.time()-start_time, 'Target:', cur_target)
    

        # save the errors
        df_save['MAE_' + cur_target] = save_mae
        df_save.to_csv('results/pred_result_' + MODEL + '_' + str(APPLY_INTERPOLATION) + '.csv', index=None)