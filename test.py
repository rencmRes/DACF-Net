import argparse
import os

import options.options as option
import utils.util as util
from data import create_dataloader, create_dataset
from models import create_model


def main():
    parser = argparse.ArgumentParser(description='Test DVCF-Net')
    parser.add_argument('-opt', default='options/test/test_dvcf.json',
                        help='Path to the test JSON file')
    opt = option.dict_to_nonedict(option.parse(parser.parse_args().opt, is_train=False))

    result_root = opt['path']['results_root']
    util.mkdir(result_root)
    model = create_model(opt)

    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_loader = create_dataloader(create_dataset(dataset_opt), dataset_opt)
        save_dir = os.path.join(result_root, dataset_opt['name'])
        util.mkdir(save_dir)

        for data in test_loader:
            need_hr = data.get('HR') is not None
            model.feed_data(data, need_HR=need_hr)
            model.test()
            sr_img = util.tensor2img(model.get_current_visuals(need_HR=need_hr)['SR'])

            image_name = os.path.splitext(os.path.basename(data['LR_path'][0]))[0]
            suffix = opt['suffix'] or ''
            util.save_img(sr_img, os.path.join(save_dir, image_name + suffix + '.tif'))


if __name__ == '__main__':
    main()
