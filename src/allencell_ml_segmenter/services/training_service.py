# import hydra
# from aics_im2im.train import entry_point_call
# from hydra.core.global_hydra import GlobalHydra
# from hydra.core.hydra_config import HydraConfig
#
#
# class TrainingService:
#     def __init__(self):
#         pass
#
#     def start_training(self):
#         GlobalHydra.instance().clear()
#         hydra.initialize(
#             version_base="1.3", config_path="../../../../aics-im2im/configs"
#         )
#
#         cfg = hydra.compose(
#             config_name="train",
#             overrides=[
#                 "trainer=cpu",
#                 "experiment=im2im/segmentation.yaml",
#                 "hydra.runtime.cwd=.",
#             ],
#             return_hydra_config=True,
#         )
#
#         HydraConfig().cfg = cfg
#         entry_point_call(cfg)
#         print(cfg.paths.output_dir)
#
#     def make_prediction(self):
#         pass
#
#     def stop_training(self):
#         pass
#
#     def continue_training(self):
#         pass
