import os
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver_vgg import Solver as Solver_VGG
from solver_resnet152 import Solver as Solver_ResNet152


def main(args):
    cudnn.benchmark = True

    load_mode = 0
    data_path = 'dataset/'
    saved_path = './npy_img/'
    save_path = './save/'
    test_patient = 'L506'
    result_fig = True
    norm_range_min = -1024.0
    norm_range_max = 3072.0
    trunc_min = -160.0
    trunc_max = 240.0
    transform = False
    batch_size = 16
    num_epochs = 60
    print_iters = 50
    decay_iters = 3000
    save_iters = 3000
    n_d_train = 4
    lr = 1e-6
    lambda_ = 10 
    device = 'cuda'
    num_workers = 10
    multi_gpu = False

    patch_n = 8
    patch_size = 120

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('Create path : {}'.format(save_path))


    fig_path = os.path.join(save_path, 'fig')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        print('Create path : {}'.format(fig_path))

    data_loader = get_loader(mode=args.mode,
                             load_mode=load_mode,
                             saved_path=saved_path,
                             test_patient=test_patient,
                             img = args.img,
                             patch_n=(None),
                             patch_size=(None),
                             transform=transform,
                             batch_size=(1),
                             num_workers=num_workers)

    if args.backbone == 'vgg19':
        model_name = 'experiment_3'
        solver = Solver_VGG(args.mode, load_mode, data_loader, device, norm_range_min, norm_range_max, trunc_min, 
                            trunc_max, save_path, multi_gpu, num_epochs, print_iters, decay_iters, save_iters, 
                            model_name, result_fig, n_d_train, patch_n, batch_size, patch_size, lr, lambda_)
    elif args.backbone == 'resnet152':
        model_name = 'experiment_2'
        solver = Solver_ResNet152(args.mode, load_mode, data_loader, device, norm_range_min, norm_range_max, trunc_min, 
                                  trunc_max, save_path, multi_gpu, num_epochs, print_iters, decay_iters, save_iters, 
                                  model_name, result_fig, n_d_train, patch_n, batch_size, patch_size, lr, lambda_)

    if args.mode == 'test':
        solver.test()
    elif args.mode == 'demo':
        solver.test()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='test', help="test | demo")
    parser.add_argument('--backbone', type=str, default='vgg19', help="vgg19 | resnet152")
    parser.add_argument('--img', type=str, default='1', help="Image to be tested in demo")

    args = parser.parse_args()
    print(args)
    main(args)
