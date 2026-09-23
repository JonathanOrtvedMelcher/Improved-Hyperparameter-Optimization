import numpy as np

total_dict = {}
found_one = False
back_statement = False
for method in ['grid_rand', 'halton', 'lhs', 'ran', 'reg', 'rsa', 'sobol', 'true_hyper']:
    for d in range(2, 5):
        for res in range(2, 6):
            combined_batch_min_loss = []
            for batch_number in range(32):
                data = np.array(np.load(f'new_data/LGBM/d_{d}_res_{res}/{method}_d_{d}_res_{res}_batch{batch_number}.p', allow_pickle=True),dtype=object)
                if len(data) != 8:
                    print('Data shape for method {}, d {}, res {}, batch {} is not 8, but {}'.format(method, d, res, batch_number, len(data)))
                min_loss_list = []
                for i in data:
                    try: 
                        loss = np.array(i)[:, -1]
                    except Exception as e:
                        print('Error processing data for method {}, d {}, res {}, batch {}: {}'.format(method, d, res, batch_number, e))
                        continue
                    number_of_points = len(loss)
                    min_loss = np.min(loss)
                    combined = np.array([batch_number, number_of_points, min_loss])
                    min_loss_list.append(combined)
                combined_batch_min_loss.append(min_loss_list)
            combined_batch_min_loss = np.array(combined_batch_min_loss, dtype=np.float64)
            total_dict[(method, d, res)] = np.concatenate(combined_batch_min_loss)

import pickle
with open('total_dict_full.pkl', 'wb') as f:
    pickle.dump(total_dict, f)