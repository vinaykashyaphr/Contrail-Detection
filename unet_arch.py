from torch import nn
import torch
import timm
import segmentation_models_pytorch as smp


class UNet(torch.nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.model = smp.Unet(encoder_name=config['encoder_name'],
                        encoder_weights=config['encoder_weights'],
                        decoder_use_batchnorm=config['batchnorm_usage'],
                        classes=config['number_of_classes']
                        )

    def forward(self, images):
        predictions = self.model(images)
        return predictions


class EncoderModule(nn.Module):
    def __init__(self):
        super(EncoderModule, self).__init__()
        self.encoder = timm.create_model("resnest101e", pretrained=True)
        self.stages = nn.ModuleList(
            [
                nn.Identity(),
                nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.act1),
                nn.Sequential(self.encoder.maxpool, self.encoder.layer1),
                self.encoder.layer2,
                self.encoder.layer3,
                self.encoder.layer4,
            ]
        )

    def forward(self, x):
        features = []
        for feature in self.stages:
            x = feature(x)
            features.append(x)
        return features


# features: single conv block, double convtranspose layers, upconv-upsample
class UnetModule(nn.Module):
    def __init__(self, upmapping="upconv"):
        super(UnetModule, self).__init__()
        self.upmapping = upmapping
        self.encoder = EncoderModule()
        self.dec_in_c = [2048, 256, 128, 64, 32]
        self.dec_out_c = [256, 128, 64, 32, 16]
        self.skip_c = [1024, 512, 256, 128, 0]

        self.module_list = nn.ModuleList()
        for i in range(len(self.dec_in_c)):
            if upmapping == "upsample":
                act_channels = self.dec_in_c[i]
            else:
                act_channels = self.dec_in_c[i] // 2
            self.module_list.append(
                nn.ModuleList(
                    [
                        self.expanding_unit(
                            self.dec_in_c[i], self.dec_in_c[i] // 2, 2, 0
                        ),
                        self.base_unit(
                            act_channels + self.skip_c[i], self.dec_out_c[i], 3, 1
                        ),
                    ]
                )
            )

        self.final_conv = nn.Conv2d(self.dec_out_c[-1], 1, kernel_size=1)

    def base_unit(self, in_c, out_c, f, p):
        return nn.Sequential(
            nn.Dropout(0.4),
            nn.Conv2d(in_c, out_c, kernel_size=f, padding=p),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(),
        )

    def expanding_unit(self, in_c, out_c, f, p):
        if self.upmapping == "upsample":
            return nn.Upsample(scale_factor=2, mode="nearest")
        else:
            return nn.Sequential(
                nn.Dropout(0.4),
                nn.ConvTranspose2d(in_c, out_c, f, padding=p, stride=2),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(),
                nn.Dropout(0.4),
                nn.ConvTranspose2d(out_c, out_c, 1, stride=1, padding=0),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(),
            )

    def center_crop(self, encoder_out, decoder_out):
        crop_dims = (
            (encoder_out.size(2) - decoder_out.size(2)) // 2,
            (encoder_out.size(3) - decoder_out.size(3)) // 2,
        )
        cropped_encoder_out = encoder_out[
            :,
            :,
            crop_dims[0] : crop_dims[0] + decoder_out.size(2),
            crop_dims[1] : crop_dims[1] + decoder_out.size(3),
        ]
        return cropped_encoder_out

    def forward(self, x):
        encoder = self.encoder(x)
        features, x = encoder[1:][:-1][::-1], encoder[1:][-1]
        for i, module in enumerate(self.module_list):
            x = module[0](x)
            if i != len(self.module_list) - 1:
                # crop = self.center_crop(features[i], x)
                x = torch.cat([x, features[i]], 1)
            x = module[1](x)
        x = self.final_conv(x)
        return x


class Unet3Plus:
    pass


class AttentionUnet:
    pass
