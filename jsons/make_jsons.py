import json
import glob
import sys, os


def make_mock_data_jsons(posterior_samples_path, fname):
    '''
    Expected posterior samples name: gw_xxxxx.h5
    '''

    post_dict = {}
    files = sorted( glob.glob(posterior_samples_path + '/*.h5') )
    for i, file in enumerate(files):
        gw_name = file.split('/')[-1].split('_')[-1].split('.')[0]
        post_dict[gw_name] = file

    outpath = os.path.join(sys.path[0], fname)
    with open(outpath, 'w') as outfile:
        json.dump(post_dict, outfile)
    
    return outpath


def make_hyperparam_jsons():  # TODO
    NotImplemented


if __name__ == '__main__':
    _ = make_mock_data_jsons(posterior_samples_path='/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/lumdistsig0.01_from_small_zero_zerr_catalog', fname='lumdistsig0.01_from_1k_small_zero_zerr_catalog.json')