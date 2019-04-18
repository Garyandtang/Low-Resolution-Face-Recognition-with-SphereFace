import logging
import nerual_net
import torch
import lfw_verification_for_validate
import common_args
logging.basicConfig(filename='FYP_Resnet_validation.log', level=logging.INFO)
model_root = '/home/tangjiawei/project/fyp/saved/30000_net_backbone.pth'
if __name__ == '__main__':
    downsampling_factor = 7
    args = common_args.get_args()
    device = torch.device(1)
    print(device)

    # Setup network Spherenet20_lR
    num_layers = 20
    lr_spherenet_test = nerual_net.spherenet(num_layers, args.feature_dim, args.use_pool, args.use_dropout)
    lr_spherenet_test.load_state_dict(
        torch.load('/home/tangjiawei/PycharmProjects/FYP_Face_Verification/saved/2000_lr_spherenet_df9.pth'))
    lr_spherenet_test.to(device)

    for i in range(16):
        lfw_verification_for_validate.run(lr_spherenet_test, feature_dim=512, device=device, N=(i + 1))

