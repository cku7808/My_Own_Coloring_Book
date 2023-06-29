from flask import send_file, Flask, render_template, request

import random
import pickle

from skimage.color import rgb2lab


import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms


device = "cuda" if torch.cuda.is_available() else "cpu"
import cv2
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import tempfile

##################################################################################3
# color transform code
###################################################################################3
from functools import partial
class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size

conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.InstanceNorm2d(out_channels))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None


    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )

class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()

        # convolutional
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.norm1_1 = nn.InstanceNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # residual blocks
        self.res1 = ResNetLayer(64, 128, block=ResNetBasicBlock, n=1)
        self.res2 = ResNetLayer(128, 256, block=ResNetBasicBlock, n=1)
        self.res3 = ResNetLayer(256, 512, block=ResNetBasicBlock, n=1)

    def forward(self, x):
        x = F.relu(self.norm1_1(self.conv1_1(x)))
        c4 = self.pool1(x)
        c3 = self.res1(c4)
        c2 = self.res2(c3)
        c1 = self.res3(c2)
        return c1, c2, c3, c4

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(out_channels),
    )

class RecoloringDecoder(nn.Module):
    # c => (bz, channel, h, w)
    # [Pt, c1]: (18 + 512) -> (256)
    # [c2, d1]: (256 + 256) -> (128)
    # [Pt, c3, d2]: (18 + 128 + 128) -> (64)
    # [Pt, c4, d3]: (18 + 64 + 64) -> 64
    # [Illu, d4]: (1 + 64) -> 3

    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up_4 = double_conv(15 + 512, 256)
        self.dconv_up_3 = double_conv(256 + 256, 128)
        self.dconv_up_2 = double_conv(15 + 128 + 128, 64)
        self.dconv_up_1 = double_conv(15 + 64 + 64, 64)
        self.conv_last = nn.Conv2d(1 + 64, 3, 3, padding=1)


    def forward(self, c1, c2, c3, c4, target_palettes_1d, illu):
        bz, h, w = c1.shape[0], c1.shape[2], c1.shape[3]
        target_palettes = torch.ones(bz, 15, h, w).float().to(device)
        target_palettes = target_palettes.reshape(h, w, bz * 15) * target_palettes_1d
        target_palettes = target_palettes.permute(2, 0, 1).reshape(bz, 15, h, w)

        # concatenate target_palettes with c1
        x = torch.cat((c1.float(), target_palettes.float()), 1)
        x = self.dconv_up_4(x)
        x = self.upsample(x)

        # concatenate c2 with x
        x = torch.cat([c2, x], dim=1)
        x = self.dconv_up_3(x)
        x = self.upsample(x)

        # concatenate target_palettes and c3 with x
        bz, h, w = x.shape[0], x.shape[2], x.shape[3]
        target_palettes = torch.ones(bz, 15, h, w).float().to(device)
        target_palettes = target_palettes.reshape(h, w, bz * 15) * target_palettes_1d
        target_palettes = target_palettes.permute(2, 0, 1).reshape(bz, 15, h, w)
        x = torch.cat([target_palettes.float(), c3, x], dim=1)
        x = self.dconv_up_2(x)
        x = self.upsample(x)

        # concatenate target_palettes and c4 with x
        bz, h, w = x.shape[0], x.shape[2], x.shape[3]
        target_palettes = torch.ones(bz, 15, h, w).float().to(device)
        target_palettes = target_palettes.reshape(h, w, bz * 15) * target_palettes_1d
        target_palettes = target_palettes.permute(2, 0, 1).reshape(bz, 15, h, w)
        x = torch.cat([target_palettes.float(), c4, x], dim=1)
        x = self.dconv_up_1(x)
        x = self.upsample(x)
        illu = illu.view(illu.size(0), 1, illu.size(1), illu.size(2))
        x = torch.cat((x, illu), dim = 1)
        x = self.conv_last(x)
        return x

def get_illuminance(img):
    """
    Get the luminance of an image. Shape: (h, w)
    """
    img = img.permute(1, 2, 0)  # (h, w, channel)
    img = img.numpy()
    img = img.astype(float) / 255.0 #EIDT float!!
    img_LAB = rgb2lab(img)
    img_L = img_LAB[:,:,0]  # luminance  # (h, w)
    return torch.from_numpy(img_L)

# pre-processsing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    #transforms.Resize((300, 300)),
    transforms.ToTensor(),
])
###################################################################################3
app = Flask(__name__)
# 모델 로드
model = tf.keras.models.load_model('mod_new.h5')
import tensorflow_hub as hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# 신경전이 모델 이미지 전처리
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# 임시 폴더 생성
temp_dir = tempfile.mkdtemp()
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)

# 웹 페이지 렌더링
@app.route('/')
def home():
    return render_template('index.html')


# 이미지 변환 및 다운로드(신경전이로 화풍 변환 및 스케치 추출)
@app.route('/convert', methods=['POST'])
def convert():

        # 이미지 파일 받기
        image_file = request.files['image']

        # 업로드된 파일을 임시 파일에 저장

        image_file.save(temp_dir+"/ori.png")
        print(temp_dir+"/ori.png")

        #image_array = np.frombuffer(image_data, np.uint8)
        ori_image = load_img(temp_dir+"/ori.png")

        style_image = load_img('factoryathortadeebro.jpg')
        stylized_image = hub_model(tf.constant(ori_image), tf.constant(style_image))[0]
        save_img = tensor_to_image(stylized_image)
        save_img.save(temp_dir +"/trans.png")

        new_size = (512, 512)
        img = cv2.imread(temp_dir +"/trans.png", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, new_size)
        decoded_image = np.expand_dims(img, axis=0)
        image = decoded_image.reshape(-1, 512, 512, 1)

        # 예측 수행
        prediction = model.predict(image)
        prediction = prediction.reshape(512, 512, 1)


        # 변환된 이미지를 PIL 이미지 객체로 변환
        transformed_img = np.squeeze(prediction)  # 배치 차원 제거
        transformed_img = Image.fromarray(np.uint8(transformed_img))

        # 변환된 이미지를 다운로드할 수 있도록 바이트 스트림으로 변환
        output = io.BytesIO()
        transformed_img.save(output, format='PNG')
        output.seek(0)


        # 변환된 이미지 다운로드
        return send_file(output, mimetype='image/png', as_attachment=True, download_name='transedge_image.png')



# 이미지 변환 및 다운로드(신경전이 스타일로 변환된 것의 채색을 변환)
@app.route('/transcolor', methods=['POST'])
def trans():

        color = ['1.pkl', '2.pkl', '3.pkl', '4.pkl', '5.pkl']
        random_color = random.choice(color)
        # load model from saved model file
        state_path = 'best_FE_RD.pth'
        img_path = temp_dir+"/trans.png"
        pal_path = random_color

        state = torch.load(str(state_path), map_location=torch.device('cpu'))
        FE = FeatureEncoder().float().to(device)
        RD = RecoloringDecoder().float().to(device)

        optimizer = torch.optim.AdamW(list(FE.parameters()) + list(RD.parameters()), lr=0.0002, weight_decay=4e-3)

        FE.load_state_dict(state['FE'])
        RD.load_state_dict(state['RD'])
        optimizer.load_state_dict(state['optimizer'])

        ########

        ori_img_load = transform(cv2.imread(str(img_path)))
        ori_image = torch.unsqueeze(ori_img_load, dim=0)

        illu_load = get_illuminance(ori_img_load)
        illu = torch.unsqueeze(illu_load, dim=0)

        ori_image = ori_image.double()
        illu = illu.double()

        new_palette_load = pickle.load(open(str(pal_path), 'rb'))
        new_palette_load = new_palette_load[:, :5, :].ravel() / 255.0
        new_palette_load = torch.from_numpy(new_palette_load).double()
        new_palette = torch.unsqueeze(new_palette_load, dim=0)

        ######
        flat_palette = new_palette.flatten()
        c1, c2, c3, c4 = FE.forward(ori_image.float().to(device))
        out = RD.forward(c1, c2, c3, c4, flat_palette.float().to(device), illu.float().to(device))
        new_out = np.squeeze(out, axis=0)
        new_out = new_out.detach().cpu().numpy()
        new_out = np.transpose(new_out, (1, 2, 0))
        new_out = (new_out * 255).astype(np.uint8)

        image = Image.fromarray(new_out.astype(np.uint8))
        filename = 'colortrans.png'
        image.save(filename)

        # 파일 다운로드
        return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=9000)


