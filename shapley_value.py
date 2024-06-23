import pandas as pd
import numpy as np
import os
import shap
import time
from load_data import *
from model import *
from main import *
import warnings
warnings.filterwarnings('ignore')

MODEL = 'Transformer'       # LSTM or Transformer or Linear or MovingAverage
TEST_START = 2001           # only test the model for the last year that has depression data
TARGET_STATE = 'Texas'


if __name__ == '__main__':
    start_time = time.time()
    if not os.path.exists('results/shapley'):
        os.mkdir('results/shapley')


    df_save = pd.DataFrame()
    df_save['Year'] = list(np.arange(TEST_START, 2022, 2))

    for cur_target in TARGETS:
        
        # load data
        dict_data = load_data_macro(time_step=TIME_STEP, target=cur_target, state=TARGET_STATE)
        dict_train_data = load_data(time_step=TIME_STEP, target=cur_target)

        for year in range(TEST_START, 2022, 2):
            cur_time = time.time()

            train_x, _, _, _, _, _ = split_data(dict_train_data, year)

            test_x = dict_data[str(year) + '_feat']
            test_x = np.transpose(test_x, (0, 2, 1))
            test_x = torch.from_numpy(test_x).to(DEVICE)

            # load model
            model = torch.load('results/models/' + MODEL + '_' + str(year) + '.model',map_location=torch.device('cpu')).to(DEVICE)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

            # model evaluation
            model.eval()
            pred = model(test_x)

            # create an explainer to calculate shapley value
            explainer = shap.GradientExplainer(model, train_x)
            shap_values = explainer.shap_values(test_x)
            print(year, 'Total Time Cost:', time.time()-start_time, 'Target:', cur_target, 'SHAP:', shap_values.shape)
            shap_values = np.mean(shap_values, axis=0)[:,:,0]
            df_shapley = pd.DataFrame()
            for i in range(TIME_STEP):
                df_shapley[str(i)] = shap_values[i]
            df_shapley.to_csv('results/shapley/' + MODEL + '_' + cur_target[:4] + '_' + str(year) + '_' + TARGET_STATE + '.csv', index=None)