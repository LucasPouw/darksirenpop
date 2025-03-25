import json
import glob
import sys, os


def make_mock_data_jsons(posterior_samples_path):
    '''
    Expected posterior samples name: gw_xxxxx.h5
    '''

    post_dict = {}
    for file in sorted( glob.glob(posterior_samples_path + '/*.h5') ):
        gw_name = file.split('/')[-1].split('_')[-1].split('.')[0]
        post_dict[gw_name] = file

    with open(os.path.join(sys.path[0], 'posterior_samples_mock_v6.json'), 'w') as outfile:
        json.dump(post_dict, outfile)


def make_hyperparam_jsons():  # TODO
    NotImplemented


if __name__ == '__main__':
    make_mock_data_jsons(posterior_samples_path='/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/mock_posteriors_v6')