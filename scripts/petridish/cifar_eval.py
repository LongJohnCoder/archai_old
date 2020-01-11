from FastAutoAugment.common.common import common_init
from FastAutoAugment.nas.evaluate import eval_arch
from FastAutoAugment.petridish.petridish_micro_builder import PetridishMicroBuilder

if __name__ == '__main__':
    conf = common_init(config_filepath='confs/petridish_cifar.yaml',
                       experiment_name='petridish_cifar_eval')

    conf_eval = conf['nas']['eval']

    # evaluate architecture using eval settings
    eval_arch(conf_eval, micro_builder=PetridishMicroBuilder())

    exit(0)

