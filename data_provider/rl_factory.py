from data_provider.rl_loader import MSLRLLoader, PSMRLLoader, SMAPRLLoader, SMDRLLoader, SWATRLLoader
from torch.utils.data import DataLoader

data_dict = {
        'PSM': PSMRLLoader,
        'MSL': MSLRLLoader,
        'SMAP': SMAPRLLoader,
        'SMD': SMDRLLoader,
        'SWaT': SWATRLLoader,
    }

def data_provider(args, setting, flag):
    Data = data_dict[args.data]
    batch_size = args.batch_rl
    drop_last = False
    data_set = Data(
        root_path=args.root_path,
        setting = setting,
        win_size=args.win_size,
        flag=flag
    )
    print(f"{flag} Length of Loader {len(data_set)}")
    data_loader = DataLoader(
        data_set,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
        drop_last=drop_last,
        )
    return data_set, data_loader
