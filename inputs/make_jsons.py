import json
import glob
import sys, os


def make_mock_data_jsons(posterior_samples_path, skymaps_path):
    '''
    Expected posterior samples name: gw_xxxxx.h5
    Expected skymap name: skymap_*.h5  TODO: add zero-padding to skymap names...
    '''

    skymap_dict = {}
    for file in sorted( glob.glob(skymaps_path + '/*.fits') ):
        gw_name = file.split('/')[-1].split('_')[-1].split('.')[0]
        skymap_dict[gw_name] = file

    post_dict = {}
    for file in sorted( glob.glob(posterior_samples_path + '/*.h5') ):
        gw_name = file.split('/')[-1].split('_')[-1].split('.')[0]
        if gw_name not in skymap_dict.keys():
            print(f'{gw_name} has no skymap, skipping')
            continue
        post_dict[gw_name] = file


    with open(os.path.join(sys.path[0], 'skymaps_mock.json'), 'w') as outfile:
        json.dump(skymap_dict, outfile)

    with open(os.path.join(sys.path[0], 'posterior_samples_mock.json'), 'w') as outfile:
        json.dump(post_dict, outfile)


if __name__ == '__main__':
    make_mock_data_jsons(posterior_samples_path='/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/mock_posterior_samples',
                         skymaps_path='/net/vdesk/data2/pouw/MRP/mockdata_analysis/darksirenpop/output/mock_skymaps')