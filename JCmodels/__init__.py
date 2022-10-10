# from utils.registry import MODEL_REGISTRY
# from utils.misc import scandir
# import importlib
# import os.path as osp



# __all__ = ['build_model']
# # automatically scan and import model modules for registry
# # scan all the files under the 'models' folder and collect files ending with
# # '_model.py'
# model_folder = osp.dirname(osp.abspath(__file__))
# model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]

# # import all the model modules
# _model_modules = [importlib.import_module(f'models.{file_name}') for file_name in model_filenames]


# def build_model(network_kwargs):
#     """Build model from options.

#     Args:
#         opt (dict): Configuration. It must contain:
#             model_type (str): Model type.
#     """
#     model = MODEL_REGISTRY.get(network_kwargs['model_type'])(**network_kwargs)
#     return model