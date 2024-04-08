def dataset_type(x):
    if str(type(x)).__contains__("tensorflow"):
        return "tensorflow_dataset"
    elif str(type(x)).__contains__("DataLoader"):
        return "torch_dataloader"
    # elif str(type(x)).__contains__("numpy.ndarray"):
    #     return "numpy_ndarray"
    else:
        return "not_supported"
