from FastAutoAugment.common.common import common_init
from FastAutoAugment.random_arch.random_micro_builder import RandomMicroBuilder
from FastAutoAugment.nas.model_desc import RunMode
from FastAutoAugment.nas import nas_utils

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/random_cifar.yaml',
                       param_args=['--common.experiment_name', 'random_cifar_search'])

    # region config
    conf_search = conf['nas']['search']
    conf_model_desc = conf_search['model_desc']
    final_desc_filename = conf_search['final_desc_filename']
    # endregion

    # create model and save it to yaml
    # NOTE: there is no search here as the models are just randomly sampled
    model_desc = nas_utils.create_model_desc(conf_model_desc,
                                             run_mode=RunMode.Search,
                                             micro_builder=RandomMicroBuilder())

    # save model to location specified by eval config
    model_desc.save(final_desc_filename)

    exit(0)