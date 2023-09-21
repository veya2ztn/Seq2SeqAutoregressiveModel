from trainer import main_worker
from configs.arguments import get_args
from utils.tools import find_free_port
import torch
from configs.utils import print_namespace_tree
from dataset.get_resource import create_memory_templete
def distributed_initial(args):
    import os
    ngpus = ngpus_per_node                          = torch.cuda.device_count()
    args.Pengine.engine.world_size                  = -1
    args.Pengine.engine.dist_file                   = None
    args.Pengine.engine.rank                        = 0
    args.Pengine.engine.dist_backend                = "nccl"
    args.Pengine.engine.multiprocessing_distributed = ngpus>1
    args.Pengine.engine.ngpus_per_node              = ngpus_per_node
    if not hasattr(args,'train_set'):args.Pengine.engine.train_set='large'
    ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = find_free_port()#os.environ.get("MASTER_PORT", f"{find_free_port()}" )
    args.Pengine.engine.port = port
    args.Pengine.engine.dist_url = f"tcp://{ip}:{port}"
    if args.Pengine.engine.world_size == -1 and "SLURM_NPROCS" in os.environ:
        args.Pengine.engine.world_size = int(os.environ["SLURM_NPROCS"])
        args.Pengine.engine.rank       = int(os.environ["SLURM_PROCID"])
        jobid           = os.environ["SLURM_JOBID"]

        hostfile        = "dist_url." + jobid  + ".txt"
        if args.Pengine.engine.dist_file is not None:
            args.Pengine.engine.dist_url = "file://{}.{}".format(os.path.realpath(args.Pengine.engine.dist_file), jobid)
        elif args.Pengine.engine.rank == 0:
            import socket
            ip = socket.gethostbyname(socket.gethostname())
            port = find_free_port()
            args.Pengine.engine.dist_url = "tcp://{}:{}".format(ip, port)
            #with open(hostfile, "w") as f:f.write(args.Pengine.engine.dist_url)
        else:
            import os
            import time
            while not os.path.exists(hostfile):
                time.sleep(1)
            with open(hostfile, "r") as f:
                args.Pengine.engine.dist_url = f.read()
        print("dist-url:{} at PROCID {} / {}".format(args.Pengine.engine.dist_url, args.Pengine.engine.rank, args.Pengine.engine.world_size))
    else:
        args.Pengine.engine.world_size = 1
    args.Pengine.engine.distributed = args.Pengine.engine.world_size > 1 or args.Pengine.engine.multiprocessing_distributed
    return args

def distributing_main(args=None):
    
    if args is None:args = get_args()
    assert args.Pengine.engine.name == 'naive_distributed'
    args = distributed_initial(args)

    train_dataset_tensor,valid_dataset_tensor,train_record_load,valid_record_load = create_memory_templete(args)
    result_tensor = torch.zeros(1).share_memory_()
    if args.Pengine.engine.multiprocessing_distributed:
        print("======== entering  multiprocessing train ==========")
        args.Pengine.engine.world_size = args.Pengine.engine.ngpus_per_node * args.Pengine.engine.world_size
        torch.multiprocessing.spawn(main_worker, nprocs=args.Pengine.engine.ngpus_per_node, args=(args.Pengine.engine.ngpus_per_node, args,result_tensor,
                                    train_dataset_tensor,train_record_load,
                                    valid_dataset_tensor,valid_record_load))
    else:
        print("======== entering  single gpu train ==========")
        main_worker(0, args.Pengine.engine.ngpus_per_node, args,result_tensor, train_dataset_tensor,train_record_load,valid_dataset_tensor,valid_record_load)
    return result_tensor


if __name__ == '__main__':
    distributing_main()
