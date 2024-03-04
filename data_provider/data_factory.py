from data_provider.data_loader import SMAPSegLoader, SMDSegLoader, MSLSegLoader, PSMSegLoader, SWATSegLoader
from torch.utils.data import DataLoader

data_dict = {
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWaT': SWATSegLoader,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    # Data = data_dict["SMAP"]
    shuffle_flag = True if flag != "test" else False
    batch_size = args.batch_size
    drop_last = False
    data_set = Data(
        root_path=args.root_path,
        win_size=args.win_size,
        flag=flag,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
        )
    return data_set, data_loader
