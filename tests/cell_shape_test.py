# import torch

# from FastAutoAugment.common.common import common_init
# from FastAutoAugment.darts.darts_micro_builder import DartsMicroBuilder
# from FastAutoAugment.nas.nas_utils import create_model

# def test_cell_shape():
#     conf = common_init(config_filepath=None,
#                        param_args=['--common.logdir', '',
#                                    '--common.experiment_name', 'unit_test'])

#     conf_data = conf['dataset']
#     conf_search = conf['nas']['search']
#     conf_model_desc = conf_search['model_desc']

#     micro_builder = DartsMicroBuilder()
#     device = torch.device('cuda')

#     model = create_model(conf_model_desc, device,
#                          aux_tower=False, affine=False, droppath=False,
#                          micro_builder=None, template_model_desc=None)

#     x = torch.randn(64, 3, 32, 32).to(device)

#     s0 = model._stem0_op(x)
#     s1 = model._stem1_op(x)

#     assert list(s0.shape) == [64, 48, 32, 32]

#     logits_aux = None
#     for cell in model._cells:
#         s0, s1 = s1, cell.forward(s0, s1)
#         if cell.aux_tower is not None:
#             logits_aux = cell.aux_tower(s1)

#     # s1 is now the last cell's output
#     out = model.final_pooling(s1)
#     logits = model.linear(out.view(out.size(0), -1))  # flatten
