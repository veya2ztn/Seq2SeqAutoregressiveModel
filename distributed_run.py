from trainer import main_worker
from utils.params import get_args
from utils.tools import distributed_initial
import torch

def create_memory_templete(args):
    train_dataset_tensor = valid_dataset_tensor = train_record_load = valid_record_load = None
    if args.use_inmemory_dataset:
        assert args.dataset_type
        print("======== loading data as shared memory==========")
        if not ('fourcast' in args.mode):
            print(f"create training dataset template, .....")
            train_dataset_tensor, train_record_load = eval(args.dataset_type).create_offline_dataset_templete(split='train' if not args.debug else 'test',
                                                                                                              root=args.data_root, use_offline_data=args.use_offline_data, dataset_flag=args.dataset_flag)
            train_dataset_tensor = train_dataset_tensor.share_memory_()
            train_record_load = train_record_load.share_memory_()
            print(
                f"done! -> train template shape={train_dataset_tensor.shape}")

            print(f"create validing dataset template, .....")
            valid_dataset_tensor, valid_record_load = eval(args.dataset_type).create_offline_dataset_templete(split='valid' if not args.debug else 'test',
                                                                                                              root=args.data_root, use_offline_data=args.use_offline_data, dataset_flag=args.dataset_flag)
            valid_dataset_tensor = valid_dataset_tensor.share_memory_()
            valid_record_load = valid_record_load.share_memory_()
            print(
                f"done! -> train template shape={valid_dataset_tensor.shape}")
        else:
            print(f"create testing dataset template, .....")
            train_dataset_tensor, train_record_load = eval(args.dataset_type).create_offline_dataset_templete(split='test',
                                                                                                              root=args.data_root, use_offline_data=args.use_offline_data, dataset_flag=args.dataset_flag)
            train_dataset_tensor = train_dataset_tensor.share_memory_()
            train_record_load = train_record_load.share_memory_()
            print(f"done! -> test template shape={train_dataset_tensor.shape}")
            valid_dataset_tensor = valid_record_load = None
        print("========      done        ==========")
    return train_dataset_tensor, valid_dataset_tensor, train_record_load, valid_record_load

def distributing_main(args=None):
    
    if args is None:args = get_args()

    args = distributed_initial(args)
    
    train_dataset_tensor,valid_dataset_tensor,train_record_load,valid_record_load = create_memory_templete(args)
    result_tensor = torch.zeros(1).share_memory_()
    if args.multiprocessing_distributed:
        print("======== entering  multiprocessing train ==========")
        args.world_size = args.ngpus_per_node * args.world_size
        torch.multiprocessing.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args,result_tensor,
                                    train_dataset_tensor,train_record_load,
                                    valid_dataset_tensor,valid_record_load))
    else:
        print("======== entering  single gpu train ==========")
        main_worker(0, args.ngpus_per_node, args,result_tensor, train_dataset_tensor,train_record_load,valid_dataset_tensor,valid_record_load)
    return result_tensor


if __name__ == '__main__':
    distributing_main()
